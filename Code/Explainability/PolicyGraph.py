import time
from argparse import Namespace
import networkx as nx
from matplotlib import pyplot as plt
from collections import defaultdict
import pickle

import numpy as np
from Code.PantheonRL.overcookedgym.human_aware_rl.overcooked_ai.overcooked_ai_py.mdp.overcooked_mdp \
    import OvercookedState, PlayerState, ObjectState

# Class that computes the Policy Graph of an ego agent
from Code.Utils.Game_utils import generate_game
from Code.Utils.utils import action_num_to_char


class PolicyGraph():
    ## ENVIRONMENT PARAMETERS
    ego = None
    alt = None
    env = None
    altenv = None

    ## PLOT PARAMETERS
    FONT_SIZE = 8
    MDP_MODELS_FOLDER = 'MDP_Models/'

    ################## CONSTRUCTOR ##################

    def __init__(self, ego_file, alt_file, total_episodes, seeds, discretizer):

        self.seeds = seeds
        self.epoch_mean_time = 0

        # Metrics of the original Agent
        self.agent_metrics = {'AER': [], 'STD': []}

        # Metrics of the PG
        self.pg_metrics = {'AER': [], 'STD': []}

        self.params = Namespace(
            alt='PPO',
            alt_config={},
            alt_load=alt_file,
            device='auto',
            ego='PPO',
            ego_config={'verbose': 1},
            ego_load=ego_file,
            env='OvercookedMultiEnv-v0',
            env_config={'layout_name': 'simple'},
            framestack=1,
            record=None,
            render=False,
            seed=None,
            total_episodes=total_episodes)

        ## Discretizer
        # Used to discretize the state of the game
        self.discretizer = discretizer
        # For each predicate we assign an index, and a color
        # Example: labels = {'None-off', 'None-finished', ...}
        # Example: colors = ['blue', 'red', ...]
        self.labels, self.colors = self.discretizer.get_possible_states()
        self.actions = self.discretizer.get_possible_actions()

        ## POLICY GRAPH
        # We create a Full Directed Graph with all weights to 0
        self.pg = nx.DiGraph(name='MDP')
        # self.pg.add_weighted_edges_from([('A','B',4.0)], action='Top', color='r')

        # Frequencies: We use this histogram in order to compute well all the weights
        self.frequencies = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Generates the environment, agents...
        self.env, self.altenv, self.ego, self.alt = generate_game(self.params)

    ################## GENERATE PG ##################

    # Runs the (ego,alt) agent multiple epochs (Games) in different layouts and seeds
    # We decided to run only in the same layout where it has been trained.
    def feed(self, verbose=False):
        # layouts = LAYOUT_LIST
        layouts = ['simple']

        num_iters = len(layouts) * len(self.seeds)
        actual_iter = 1
        for seed in self.seeds:
            self.params.seed = seed

            # Generates the environment, agents...
            self.env, self.altenv, self.ego, self.alt = generate_game(self.params)

            # Play an epoch
            self.play_epoch(self.params.total_episodes, verbose=verbose)

            # Compute how much time we spent
            expected_time = time.gmtime(self.epoch_mean_time * (num_iters - actual_iter))
            expected_time = time.strftime("%H:%M:%S", expected_time)
            print(f'Iteration Completed {actual_iter}/{num_iters} ({100 * actual_iter // num_iters}%) '
                  f'Expected Time: {expected_time}')
            print('---------------------------------')

            actual_iter += 1

        # Once we finished the feeding, then we build the Graph
        self.build_mdp(verbose=verbose)
        print('\n+ End of feeding the Policy Graph.')
        print('+ Metrics of the agent over all the seeds:')
        print('\t* Average Reward:', sum(self.agent_metrics['AER']) / len(self.agent_metrics['AER']))
        print('\t* Standard Deviation:', sum(self.agent_metrics['STD']) / len(self.agent_metrics['STD']), '\n')

    # Play an Epoch in current env in order to feed the PG
    # An epoch it is formed by a number of episodes
    def play_epoch(self, num_episodes, verbose=False):
        start_time = time.time()
        rewards = []
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            reward = 0
            info = None
            while not done:
                # We get the action
                action = self.ego.get_action(obs, False)

                if info is not None:
                    obs_state = info['Obs_str']
                else:
                    obs_state = None

                # Run step and save the obs
                # obs, newreward, done, info = self.env.step(action)
                obs, newreward, done, info = self.env.step(action)
                obs_state_next = info['Obs_str']


                # Update global reward
                reward += newreward

                # Update PG with current obs and action
                # obs: Observation featurized
                # obs_state: Normal Observation
                # obs_map: Map with the state of each pos
                self.update_frequencies(obs_state, action, obs_state_next, verbose=verbose)

                if verbose:
                    obs_map = self.env.env.base_env.mdp.state_string(obs_state_next)
                    print('Mapa:\n')
                    print(obs_map)

            rewards.append(reward)

        self.env.close()

        # Compute the average reward and std
        average_reward = sum(rewards) / num_episodes
        std = np.std(rewards)
        self.agent_metrics['AER'].append(average_reward)
        self.agent_metrics['STD'].append(std)
        self.epoch_mean_time = time.time() - start_time
        # Compute how much time we spent

        print(f"Average Reward: {average_reward} and Standard Deviation: {std} --> ET: {self.epoch_mean_time}")

    # Given an observation, an action and the resulting obs, update self.frequencies correctly.
    def update_frequencies(self, obs: OvercookedState, act, next_state: OvercookedState, verbose=False):
        # Compute both predicates
        predicate = self.discretizer.get_predicates(obs)
        predicate_next = self.discretizer.get_predicates(next_state)

        if verbose:
            # From 'predicate' choosing 'act' we achieved state 'predicate_next'
            print('From', predicate, ' -> ', action_num_to_char(act), ' -> ', predicate_next)

        self.frequencies[predicate][act][predicate_next] += 1

    # Takes self.frequencies and builds the MDP
    def build_mdp2(self, verbose=False):
        for state, actions in self.frequencies.items():
            for action, next_states in actions.items():
                # List with all the frequencies of this (state, action)
                freqs_next_states = np.array([f for _, f in next_states.items()], dtype=np.int)
                # Sum of this frequencies
                sum_freqs = np.sum(freqs_next_states)
                # Convert it to Probability Distribution
                probabilities = freqs_next_states/sum_freqs
                # Sum the probability Distribution (Should be 1)
                sum_probs = np.sum(probabilities)
                # List of new edges to add with the probability

                #new_edges = [(state, next_state, prob) for next_state, _ in next_states.items() for prob in probabilities]
                new_edges = list(zip([state] * len(probabilities), next_states.keys(), probabilities))

                assert sum_probs == 1.0, 'Error: Probab. Distrib. of node {} is {}. (Should be 1.0)'.format(state,
                                                                                                            sum_freqs)
                # Add new_edges to the Graph
                print('-------')
                print(state, action, next_states.items())
                print('New Edges:', new_edges)
                print('exist edges:', [self.pg.has_edge(s,n) for s, n, p in new_edges])
                self.pg.add_weighted_edges_from(new_edges, action=action, color='blue')
                e = list(self.pg.edges)
                p = np.array([self.pg.get_edge_data(v, u)['weight'] for v, u in e])
                print('suma:', np.sum(p), 'prob:', list(zip(e,p)))
                #print(sum_probs, 'SUma', np.sum(np.array([p for _,_,p in new_edges])), new_edges)

            e = list(self.pg.edges)
            p = np.array([self.pg.get_edge_data(v,u)['weight'] for v,u in e])
            print(np.sum(p), p)

        if verbose: self.print_frequencies()

    # Takes self.frequencies and builds the MDP
    def build_mdp(self, verbose=False):
        # For each state, we take into account the most probable action
        for state, actions in self.frequencies.items():
            if verbose: print('Estem al estat:', state)
            action_keys = np.array(list(actions.keys()))
            if verbose: print('\tHem pres les accions:\t', action_keys)
            freq_actions = np.array([sum(list(actions[a].values())) for a in action_keys], dtype=np.int)
            freq_actions_sum = np.sum(freq_actions)
            if verbose: print('\t\tFreqs:\t\t\t\t', freq_actions, 'Total:', freq_actions_sum)
            prob_actions = freq_actions/freq_actions_sum
            prob_actions_sum = np.sum(prob_actions)
            if verbose: print('\t\tProbs:\t\t\t\t', prob_actions, 'Total:', prob_actions_sum)

            action_more_prob = action_keys[np.argmax(prob_actions)]
            if verbose: print('\t\tAccio més probable:', action_more_prob, 'amb prob:', prob_actions[np.argmax(prob_actions)])

            next_states = actions[action_more_prob]
            if verbose: print('\t\t\tNext states accio', action_more_prob, ':', list(next_states.items()))
            freq_next = np.array([freq for _,freq in list(next_states.items())], dtype=np.int)
            freq_next_sum = np.sum(freq_next)
            if verbose: print('\t\t\tFrequencies', freq_next, 'Total:', freq_next_sum)
            prob_next_s = freq_next / freq_next_sum
            prob_next_s_sum = np.sum(prob_next_s)
            if verbose: print('\t\t\tProbabilities', prob_next_s, 'Total:', prob_next_s_sum)

            #assert prob_next_s_sum == 1.0, 'Error: Prob. Dist. of node {} is {}.'.format(state, prob_next_s_sum)
            # Add new_edges to the Graph
            new_edges = list(zip([state]*len(list(next_states.keys())), list(next_states.keys()), prob_next_s))
            if verbose: print('Append the edges:', new_edges)
            if verbose: print('---------')
            self.pg.add_weighted_edges_from(new_edges, action=action_more_prob, color='blue')

    # Prints the attribute frequencies
    def print_frequencies(self):
        for state, actions in self.frequencies.items():
            print('State:', state)
            for action, next_states in actions.items():
                print('\t-', action)
                for next_state, freq in next_states.items():
                    print('\t\t-', next_state, freq)


    ################## TEST PG ##################
    # TODO: Tests the PG and returns it's average episode reward and std.
    # Tests the agent and computes metrics in order to compare them
    # against the orogonal RL agent
    def test(self, num_episodes=100, verbose=False):
        rewards = []
        for episode in range(num_episodes):
            self.env.reset()
            done = False
            reward = 0
            actual_state = None

            while not done:
                # We get the action
                # action = self.ego.get_action(obs, False)
                print('-------------------------------')
                predicate = self.discretizer.get_predicates(actual_state)
                action = self.select_action_using_mdp(predicate)

                _, newreward, done, info = self.env.step(action)

                if info is not None: next_state = info['Obs_str']

                # Run step and save the obs
                #print(actual_state)
                print('predicate', predicate, 'action', action_num_to_char(action))


                # Update global reward
                reward += newreward

                if verbose:
                    obs_map = self.env.env.base_env.mdp.state_string(next_state)
                    print('Mapa:\n', obs_map)

                actual_state = next_state

            rewards.append(reward)

        self.env.close()

        # Compute the average reward and std
        average_reward = sum(rewards) / num_episodes
        std = np.std(rewards)
        # self.average_rewards_agent.append(average_reward)
        # self.standard_deviations_agent.append(std)
        self.agent_metrics['AER'].append(average_reward)
        self.agent_metrics['STD'].append(std)
        print(f"Average Reward: {average_reward}")
        print(f"Standard Deviation: {std}")
        self.pg_metrics['AER'].append(0)
        self.pg_metrics['STD'].append(0)

    # FIXME: Cal arreglar aquesta funcio perque peta
    # Given a predicate, goes to the MDP and selects the corresponding action
    def select_action_using_mdp(self, predicate, num_actions=1):
        np.random.seed(time.localtime())

        # Out Edges of node 'predicate'
        outs = [(self.pg.get_edge_data(v, u)['action'], self.pg.get_edge_data(v, u)['weight']) for v, u in
                self.pg.out_edges(predicate)]

        # If there aren't out edges, then return random action
        if len(outs) == 0: return np.random.choice(self.actions, 1)[0]

        # Probability distribution
        p = [w for _, w in outs]
        a = [a for a, _ in outs]

        possible_actions = [action for action in self.actions if action not in a]

        # If sum != 1, add a new random action to the possible actions to take
        s = sum(p)
        if s < 1:
            probability_left = round(1 - s, 5)
            p.append(probability_left)
            a.append(np.random.choice(possible_actions))
            print("Hem d'afegir", probability_left)

        if s > 1:
            print('Predicate', predicate, 'Prob:', p, 'Acions:', a, 'Out Edges:', outs)
            self.show(allow_recursion=True)

        # Take one action with a given Probability distribution
        rand_action = np.random.choice(a, p=p)

        return rand_action

    # TODO: S'ha de fer tota la part de comparacio
    ################## COMPARE AGENT AGAINST PG ##################
    # Compares the metrics of the original agent vs the PG
    def compare(self):
        AER_Agent = sum(self.agent_metrics['AER']) / len(self.agent_metrics['AER'])
        STD_Agent = sum(self.agent_metrics['STD']) / len(self.agent_metrics['STD'])

        AER_PG = sum(self.pg_metrics['AER']) / len(self.pg_metrics['AER'])
        STD_PG = sum(self.pg_metrics['STD']) / len(self.pg_metrics['STD'])

        diff_AER = abs(AER_Agent - AER_PG)
        diff_STD = abs(STD_Agent - STD_PG)

        print('++++++++++++++++++++++++++++++++++++++++++++')
        print('COMPARATION')
        print('Original Agent')
        print('\t- Average Episode Reward:', AER_Agent)
        print('\t- Standard Deviation:', STD_Agent)
        print('Policy Graph')
        print('\t- Average Episode Reward:', AER_PG)
        print('\t- Standard Deviation:', STD_PG)
        print('Difference')
        print('\t- Average Episode Reward Diff:', diff_AER)
        print('\t- Standard Deviation Diff:', diff_STD)
        print('++++++++++++++++++++++++++++++++++++++++++++')

        return (diff_AER,
                diff_STD)


    ################## SHOW PG ##################
    # Plots the PG
    def show(self, allow_recursion=False):
        num_of_decimals = 2
        max_arrow_size = 4

        # In order to save it
        pos = nx.shell_layout(self.pg)

        # Save the color and label of each edge
        edge_labels = {}
        edge_colors = []
        for edge in self.pg.edges:
            if edge[0] != edge[1] or allow_recursion:
                atributes = self.pg.get_edge_data(edge[0], edge[1])
                weight = atributes['weight']
                edge_labels[edge] = '{} - {}'.format(action_num_to_char(atributes['action']),
                                               round(atributes['weight'],num_of_decimals))
                edge_colors.append(self.select_edge_color(weight))

        # Draw the Graph
        nx.draw(
            self.pg, pos,
            edge_color=edge_colors,
            #width=[self.pg.get_edge_data(v,u)['weight']*max_arrow_size for v,u in list(self.pg.edges()) if v != u],
            width=3,
            linewidths=1,
            node_size=4000,
            arrowsize=15,
            node_color='#FFD365',
            alpha=0.8,
            labels={node: str(node).replace('-', '\n')  for node in self.pg.nodes()},
            font_size=self.FONT_SIZE,
            edgelist=[edge for edge in list(self.pg.edges()) if edge[0] != edge[1] or allow_recursion]
        )

        nx.draw_networkx_edge_labels(
            self.pg, pos,
            edge_labels=edge_labels,
            font_color='#534340',
            label_pos=0.7,
            font_size=self.FONT_SIZE
        )



        # Show the graph
        plt.show()

        # Comment show to save it. Save the Graph
        # plt.savefig('labels.png')

    def select_edge_color(self, weight):
        if weight >= 0.75: return '#332FD0'
        if weight >= 0.5: return '#9254C8'
        if weight >= 0.25: return '#E15FED'
        else: return '#6EDCD9'



    ################## EXPORT/IMPORT PG ##################
    # Saves the MDP model into a gpickle file
    def save(self, file_name):
        nx.write_gpickle(self.pg, self.MDP_MODELS_FOLDER + file_name + '.gpickle')

    # Loads a MDP model
    def load(self, file_name):
        self.pg = nx.read_gpickle(self.MDP_MODELS_FOLDER + file_name + '.gpickle')


    def save_frequencies_dict(self, file_name):
        with open(self.MDP_MODELS_FOLDER + file_name + '.pickle', 'w') as handle:
            pickle.dump(self.frequencies, handle)

    def load_frequencies_dict(self, file_name):
        with open(self.MDP_MODELS_FOLDER + file_name + '.pickle', 'r') as handle:
            self.frequencies = pickle.load(handle)