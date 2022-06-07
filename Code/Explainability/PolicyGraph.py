from abc import ABC, abstractmethod
from argparse import Namespace
from collections import defaultdict

import networkx as nx
import time
import dill as pickle

import numpy as np

from Explainability.NetworkVisualizer import NetworkVisualizer
from Utils.Game_utils import generate_game
from Utils.utils import Actions, normalize
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState


class PolicyGraph(ABC):
    """
    Algorithm used in order to build an MDP from an RL agent.

    In order to feed the MDP, we execute the RL agent multiple times to build a DiGraph.
    Each environment step, we register that the RL Agent was in state 's1', took the action 'a'
    and ended up in the state 's2'.
    """

    MDP_MODELS_FOLDER = 'Code/MDP_Models/'

    def __init__(self, ego_file, alt_file, discretizer, name, layout):
        """
        :param ego_file: File where ego RL agent is saved
        :param alt_file: File where alt RL agent is saved
        :param discretizer: Which discretizer we want to use
        :param name: MDP name
        :param layout: Layout where the MDP will be trained
        """
        self.name = name
        self.layout = layout
        self.epoch_mean_time = 0

        # Metrics of the original Agent
        self.agent_metrics = {'AER': [], 'STD': []}

        # Metrics of the PG
        self.pg_metrics = {'AER': [], 'STD': [], 'new_state': 0, 'Visited States': 0}

        self.params = Namespace(
            alt='PPO',
            alt_config={},
            alt_load=alt_file,
            device='auto',
            ego='PPO',
            ego_config={'verbose': 1},
            ego_load=ego_file,
            env='OvercookedMultiEnv-v0',
            env_config={'layout_name': self.layout},
            framestack=1,
            record=None,
            render=False,
            seed=None,
            total_episodes=100)

        self.frequencies = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.pg = nx.MultiDiGraph(name='MDP')

        # Generates the environment, agents...
        self.env, self.altenv, self.ego, self.alt = generate_game(self.params, verbose=False)

        self.discretizer = discretizer(self.env)

        # For each predicate we assign an index, and a color
        self.actions = self.discretizer.get_possible_actions()

    # MDP Generation
    def feed(self, seeds, num_episodes, verbose=False):
        """
        Runs the agent and updates the MDP

        :param seeds: List of seeds
        :type seeds: list or range
        :param verbose: Prints additional information
        :param int num_episodes: Number of episodes to execute per seed
        """
        print('---------------------------------')
        print('* START FEEDING\n')

        self.agent_metrics['AER'] = []
        self.agent_metrics['STD'] = []

        layouts = [self.layout]

        num_iterations = len(layouts) * len(seeds)
        actual_iter = 1
        for seed in seeds:
            self.params.seed = seed
            self.params.total_episodes = num_episodes

            # Generates the environment, agents...
            self.env, self.altenv, self.ego, self.alt = generate_game(self.params, verbose=verbose)

            # Play an epoch
            self.play_epoch(num_episodes=num_episodes, verbose=verbose)

            # Compute how much time we spent
            expected_time = time.gmtime(self.epoch_mean_time * (num_iterations - actual_iter))
            expected_time = time.strftime("%H:%M:%S", expected_time)
            print(f'Feeding Iteration Completed {actual_iter}/{num_iterations} '
                  f'({100 * actual_iter // num_iterations}%) '
                  f'Expected Time: {expected_time}')
            print('---------\n')

            actual_iter += 1

        # Once we finished the feeding, then we build the Graph
        self.build_mdp(verbose=verbose)
        print('\t- Number of possible states:\t', self.discretizer.get_num_possible_states())
        print('\t- Number of used states:\t\t', len(self.frequencies.keys()))
        print('* END FEEDING')
        print('---------------------------------')
        print('* RL RESULTS')
        print('\t- Average Reward:', sum(self.agent_metrics['AER']) / len(self.agent_metrics['AER']))
        print('\t- Standard Deviation:', sum(self.agent_metrics['STD']) / len(self.agent_metrics['STD']), '\n')

    def play_epoch(self, num_episodes, verbose=False):
        """
        Plays the agent one epoch.

        Play an Epoch in current env in order to feed the PG.
        An epoch it is formed by a number of episodes.

        :param int num_episodes: Number of episodes to execute per seed
        :param verbose: Prints additional information
        """
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

                # Run step and save the observation
                # obs, new reward, done, info = self.env.step(action)
                obs, new_reward, done, info = self.env.step(action)
                obs_state_next = info['Obs_str']

                # Update global reward
                reward += new_reward

                # Update PG with current obs and action
                # obs: Observation featured
                # obs_state: Normal Observation
                # obs_map: Map with the state of each pos
                self.update_frequencies(obs_state, action, obs_state_next, verbose=verbose)

                if verbose:
                    obs_map = self.env.env.base_env.mdp.state_string(obs_state_next)
                    print('Reward:', new_reward)
                    print('Map in Next state:\n')
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

    def update_frequencies(self, obs: OvercookedState, act, next_state: OvercookedState, verbose=False):
        """
        Updates the attribute 'frequencies'.
        In this dictionary we save how many times the agent took act from obs to next_state

        Given an observation, an action and the resulting observation, update frequencies correctly.
        :param obs: Actual state
        :param act: Action
        :param next_state: Resulting state
        :param verbose: Prints additional information
        """
        # Compute both predicates
        _, predicate = self.discretizer.get_predicates(obs)
        _, predicate_next = self.discretizer.get_predicates(next_state)

        if verbose:
            # From 'predicate' choosing 'act' we achieved state 'predicate_next'
            print('From', predicate, ' -> ', Actions(act).name, ' -> ', predicate_next)

        self.frequencies[predicate][act][predicate_next] += 1

    @abstractmethod
    def build_mdp(self, verbose=False):
        """
        Takes frequencies and builds the Markov Decision Graph.
        """
        pass

    def print_frequencies(self):
        """
        Prints the attribute frequencies.
        """
        for state, actions in self.frequencies.items():
            print('State:', state)
            for action, next_states in actions.items():
                print('\t-', action)
                for next_state, freq in next_states.items():
                    print('\t\t-', next_state, freq)

    def normalize_node_weights(self, node):
        """

        :param node: Existent Node
        :return: Normalize edge weights of node
        """
        edges = self.pg.out_edges(node, data=True)
        weights = np.array([data['weight'] for u, v, data in edges])

        if not (sum(weights) > 0.999 and sum(weights) < 1.001):
            weights = normalize(weights)

            i = 0
            for u, v, data in edges:
                edges = self.pg.get_edge_data(u, v)
                print(edges)
                for key, value in edges.items():
                    if value['action'] == data['action']:
                        value['weight'] = weights[i]
                i += 1

        suma = sum([data['weight'] for u, v, data in self.pg.out_edges(node, data=True)])

        assert suma > 0.999 and suma < 1.001, f'Error: {suma}'

    def update_weights(self, old_weights, position, new_weight, alpha):
        # Difference between old and new value, using a learning factor
        diff = abs(old_weights[position] - new_weight) * alpha

        # Normalize weights without new_weight
        b = old_weights.copy()
        b[position] = 0
        b = normalize(b)

        for i in range(len(old_weights)):
            if i != position:
                old_weights[i] = old_weights[i] - (diff * b[i])

        old_weights[position] += diff

        return old_weights

    # MDP Testing
    def test(self, seeds, num_episodes, verbose=False):
        """
        Tests the MDP agent num_episodes for each seed.
        """
        print('---------------------------------')
        print('* START TESTING\n')

        self.pg_metrics['AER'] = []
        self.pg_metrics['STD'] = []
        self.pg_metrics['new_state'] = 0
        self.pg_metrics['Visited States'] = 0

        layouts = ['simple']

        num_iterations = len(layouts) * len(seeds)
        actual_iter = 1
        for seed in seeds:
            self.params.seed = seed

            # Generates the environment, agents...
            self.env, self.altenv, self.ego, self.alt = generate_game(self.params, verbose=verbose)

            # Play an epoch
            self.test_epoch(num_episodes=num_episodes, verbose=verbose)

            # Compute how much time we spent
            expected_time = time.gmtime(self.epoch_mean_time * (num_iterations - actual_iter))
            expected_time = time.strftime("%H:%M:%S", expected_time)
            print(f'Testing Iteration Completed {actual_iter}/{num_iterations} '
                  f'({100 * actual_iter // num_iterations}%) '
                  f'Expected Time: {expected_time}')
            print('---------\n')
            actual_iter += 1

        # Once we finished the feeding, then we build the Graph
        print('* END TESTING')
        print('---------------------------------')
        print('* MDP RESULTS')
        print('\t- Average Reward:', sum(self.pg_metrics['AER']) / len(self.pg_metrics['AER']))
        print('\t- Standard Deviation:', sum(self.pg_metrics['STD']) / len(self.pg_metrics['STD']), '\n')

    def test_epoch(self, num_episodes, verbose=False):
        """
        Plays the MDP agent 1 epoch.

        Tests the agent and computes metrics in order to compare them against the original agent
        """
        start_time = time.time()
        rewards = []
        for episode in range(num_episodes):
            self.env.reset()
            done = False
            reward = 0
            actual_state = None
            next_state = None

            while not done:
                # Get the predicate (actual state) and take an action
                predicate_dict, predicate = self.discretizer.get_predicates(actual_state)
                action = self.select_action_using_mdp(predicate, verbose=verbose)
                _, new_reward, done, info = self.env.step(action.value)

                if info is not None:
                    next_state = info['Obs_str']

                # Update global reward
                reward += new_reward

                if verbose and actual_state is not None:
                    print('Actual state:', predicate)
                    print(predicate_dict)
                    print('Action:', action)
                    obs_map = self.env.env.base_env.mdp.state_string(actual_state)
                    print('Map:\n' + obs_map)

                actual_state = next_state
                self.pg_metrics['Visited States'] += 1

            rewards.append(reward)

        self.env.close()

        # Compute the average reward and std
        average_reward = np.sum(rewards) / num_episodes
        std = np.std(rewards)

        self.pg_metrics['AER'].append(average_reward)
        self.pg_metrics['STD'].append(std)
        self.epoch_mean_time = time.time() - start_time
        print(f"Average Reward: {average_reward} and Standard Deviation: {std} --> ET: {self.epoch_mean_time}")

    @abstractmethod
    def select_action_using_mdp(self, predicate, verbose):
        pass

    def get_nearest_predicate(self, predicate, verbose=False):
        """ Returns the nearest predicate on the MDP. If already exists, then we return the same predicate. If not,
        then tries to change the predicate to find a similar state (Maximum change: 1 value).
        If we don't find a similar state, then we return None

        :param predicate: Existent or non-existent predicate in the MDP
        :return: Nearest predicate
        :param verbose: Prints additional information
        """
        predicate_space = self.discretizer.get_predicate_space()
        input_predicate = self.discretizer.str_predicate_to_dict(predicate)

        # Predicate exists in the MDP
        if self.pg.has_node(predicate):
            if verbose:
                print('NEAREST PREDICATE of existing predicate:', predicate)
            return predicate

        else:
            if verbose:
                print('NEAREST PREDICATE of NON existing predicate:', predicate)
            nearest_predicate = input_predicate.copy()
            for predicate, value in input_predicate.items():
                for possible_value in predicate_space[predicate]:
                    nearest_predicate[predicate] = possible_value
                    new_pred = self.discretizer.dict_predicate_to_str(nearest_predicate)
                    if self.pg.has_node(new_pred):
                        if verbose:
                            print('\tNEAREST PREDICATE in MDP:', new_pred)
                        return new_pred
                nearest_predicate[predicate] = value
        return None

    def compare(self):
        """
        Compares the metrics of the original RL agent vs the PG Agent.
        """
        try:
            agent_aer = sum(self.agent_metrics['AER']) / len(self.agent_metrics['AER'])
            agent_std = sum(self.agent_metrics['STD']) / len(self.agent_metrics['STD'])

            mdp_aer = sum(self.pg_metrics['AER']) / len(self.pg_metrics['AER'])
            mdp_std = sum(self.pg_metrics['STD']) / len(self.pg_metrics['STD'])

            transferred_learning = int((mdp_aer / agent_aer) * 100)
            diff_aer = -(agent_aer - mdp_aer)
            diff_std = -(agent_std - mdp_std)

            percentage_new_states = (self.pg_metrics['new_state']/self.pg_metrics['Visited States'])
            percentage_seen_states = (self.pg.number_of_nodes() / self.discretizer.get_num_possible_states())

            percentage_std = (mdp_std /agent_std)

        except ZeroDivisionError:
            raise Exception('Agent Metrics are missing! Have you loaded the model?')

        print('---------------------------------')
        print('* COMPARATIVE')
        print('\t- Original RL Agent')
        print('\t\t+ Average Episode Reward:', agent_aer)
        print('\t\t+ Standard Deviation:', agent_std)
        print('\t- Policy Graph')
        print('\t\t+ Average Episode Reward:', mdp_aer)
        print('\t\t+ Standard Deviation:', mdp_std)
        print('\t\t+ Num new states visited:', self.pg_metrics['new_state'])
        print('\t- Difference')
        print('\t\t+ Average Episode Reward Diff:', diff_aer)
        print('\t\t+ Standard Deviation Diff:', diff_std)
        print('\t- Transferred Learning:', transferred_learning, '%')
        print('\t- Percentage New states:', percentage_new_states * 100, '%')
        print('\t- Percentage Seen states:', percentage_seen_states * 100, '%')
        print('\t- Percentage STD:', percentage_std * 100, '%')

        return (diff_aer,
                diff_std, transferred_learning, percentage_new_states, percentage_seen_states, percentage_std)

    # Answering XAI: Questions 1
    def question1(self):
        """
        Answers the question: What will you do when you are in state X?
        """
        print('---------------------------------')
        print('* What will I do when I am in state X?')
        print('\t- Possible states\n')
        predicate_example = np.random.choice(list(self.pg.nodes()), 1)[0]
        print(self.discretizer.get_predicate_options(predicate_example, indentation=2))

        while True:
            predicate = input('\t- What will I do when I am in state: ')
            if not (self.discretizer.is_predicate(predicate)):
                print("\t\t!! Error: {} it's not a valid state".format(predicate))
            else:
                break

        possible_actions = self.get_possible_actions(predicate)
        print('\t\tI will take one of this actions:')
        for action, prob in possible_actions:
            print('\t\t\t+', action.name, '\tProbability:', round(prob * 100, 2), '%')

    def get_possible_actions(self, predicate):
        """ Given a predicate, get the possible actions and it's probabilities

        3 cases:

        - Predicate not in MDP but similar predicate found in MDP: Return actions of the similar predicate
        - Predicate not in MDP and no similar predicate found in MDP: Return all actions same probability
        - Predicate in MDP: Return actions of the predicate in MDP

        :param predicate: Existing or not existing predicate
        :return: Action probabilities of a given state
        """
        result = defaultdict(float)

        # Predicate not in MDP
        if predicate not in self.pg.nodes():
            # Nearest predicate not found -> Random action
            if predicate is None:
                result = {action: 1/len(Actions) for action in Actions}
                return sorted(result.items(), key=lambda x: x[1], reverse=True)

            predicate = self.get_nearest_predicate(predicate)

        # Out edges with actions [(u, v, a), ...]
        possible_actions = [(u, data['action'], v, data['weight'])
                            for u, v, data in self.pg.out_edges(predicate, data=True)]
        """
        for node in self.pg.nodes():
            possible_actions = [(u, data['action'], v, data['weight'])
                                for u, v, data in self.pg.out_edges(node, data=True)]
            s = sum([w for _,_,_,w in possible_actions])
            assert  s < 1.001 and s > 0.99, f'Error {s}'
        """
        # Drop duplicated edges
        possible_actions = list(set(possible_actions))
        # Predicate has at least 1 out edge.
        if len(possible_actions) > 0:
            for _, action, v, weight in possible_actions:
                result[Actions(action)] += weight
            return sorted(result.items(), key=lambda x: x[1], reverse=True)
        # Predicate does not have out edges. Then return all te actions with same probability
        else:
            result = {a: 1/len(Actions) for a in Actions}
            return list(result.items())

    # Answering XAI: Questions 2
    def question2(self):
        """
        Answers the question: When do you perform action X?
        """
        print('---------------------------------')
        print('* When do you perform action X?')
        print('\t- Possible actions\n')
        print(self.discretizer.get_action_options(indentation=2))

        while True:
            action = input('\t- When do you perform action: ').capitalize()
            if not (self.discretizer.is_action(action)):
                print("\t\t!! Error: {} it's not a valid action".format(action))
            else:
                break

        all_nodes, best_nodes = self.get_when_perform_action(action)
        print(f"\t\t\tPossible ({len(all_nodes)} states)\t\tMost probable ({len(best_nodes)} states)")
        for i in range(len(all_nodes)):
            if i < len(best_nodes):
                print(f"\t\t- {all_nodes[i]}\t\t{best_nodes[i]}")
            else:
                print(f"\t\t- {all_nodes[i]}")

    def get_when_perform_action(self, action):
        """ When do you perform action

        :param action: Valid action
        :return: Set of states that has an out edge with the given action
        """
        # Nodes where 'action' it's a possible action
        # All the nodes that has the same action (It has repeated nodes)
        all_nodes = [u for u, v, a in self.pg.edges(data='action') if Actions(a) == Actions[action]]
        # Drop all the repeated nodes
        all_nodes = list(set(all_nodes))

        # Nodes where 'action' it's the most probable action
        all_edges = [list(self.pg.out_edges(u, data=True)) for u in all_nodes]

        all_best_actions = [
            sorted([(u, data['action'], data['weight']) for u, v, data in edges], key=lambda x: x[2], reverse=True)[0]
            for edges in all_edges]
        best_nodes = [u for u, a, w in all_best_actions if Actions(a) == Actions[action]]

        return all_nodes, best_nodes

    # Answering XAI: Questions 3
    def question3(self):
        """
        Answers the question: Why do you perform action X in state Y?
        """
        print('---------------------------------')
        print('* Why did not you perform X action in Y state?')
        print('\t- Possible actions\n')
        print(self.discretizer.get_action_options(indentation=2))

        while True:
            action = input('\t- Why did not you perform action: ').capitalize()
            if not (self.discretizer.is_action(action)):
                print("\t\t!! Error: {} it's not a valid action".format(action))
            else:
                break

        print('\t- Possible states\n')
        predicate_example = np.random.choice(list(self.pg.nodes()), 1)[0]
        print(self.discretizer.get_predicate_options(predicate_example, indentation=2))
        while True:
            predicate = input('\t- In state: ')
            if not (self.discretizer.is_predicate(predicate)):
                print("\t\t!! Error: {} it's not a valid state".format(predicate))
            else:
                break

        # Predicate not in MDP
        if predicate not in self.pg.nodes():
            print(f"\t\t{predicate} is not in MDP. The nearest state is:")
            predicate = self.get_nearest_predicate(predicate)
            # Nearest predicate not found -> Random action
            if predicate is None:
                print(f"\t\t\t- I haven't a similar predicate. The decision had been random")

            else:
                print(f"\t\t\t- {predicate}")

        result = self.nearby_predicates(predicate)
        print(f"\t\tI have not chosen {action} in {predicate} because:")
        for a, v, diff in result:
            print(f"\t\t\t- If I chose {a.name} then I would end up to {v}")
            for predicate_key,  predicate_value in diff.items():
                print(f"\t\t\t\t- {predicate_key} --> Now: {predicate_value[0]}\tHypothetically: {predicate_value[1]}")

    def nearby_predicates(self, state):
        """
        Gets nearby states from state

        :param state: State
        :return: List of [Action, destination_state, difference]
        """
        outs = self.pg.out_edges(state, data=True)
        outs = [(u, d['action'], v, d['weight']) for u, v, d in outs]
        result = [(Actions(a), v, self.subtract_predicates(u, v)) for u, a, v, w in outs]
        return result

    def subtract_predicates(self, origin, destination):
        """
        Subtracts 2 predicates, getting only the values that are different

        :param origin: Origin predicate
        :type origin: Union[str, list]
        :param destination: Destination predicate
        :return dict: Dict with the different values
        """
        if type(origin) is str:
            origin = origin.split('-')
        if type(destination) is str:
            destination = destination.split('-')

        result = {}
        predicate_space = self.discretizer.get_predicate_space().keys()
        for value1, value2, predicate_key in zip(origin, destination, predicate_space):
            if value1 != value2:
                result[predicate_key] = (value1, value2)
        return result

    # Visualization
    def show(self, allow_recursion=False):
        """
        Normal plots for the graph
        """

        print('---------------------------------')
        print('* INTERACTIVE VISUALIZATION')
        nt = NetworkVisualizer(self.pg, layout=self.layout, name=self.name)
        nt.show(allow_recursion=allow_recursion,
                font_size=8
                )

    def show_interactive(self, show_options=False, second_display=False, subgraph=None):
        """
        Creates a html file with an interactive visualization of the graph.
        """

        print('---------------------------------')
        print('* INTERACTIVE VISUALIZATION')
        nt = NetworkVisualizer(self.pg, layout=self.layout, name=self.name)
        nt.show_interactive(frequencies=self.frequencies,
                            show_options=show_options,
                            second_display=second_display,
                            subgraph=subgraph
                            )
        print('\t- Open file:', nt.get_file_path())

    # EXPORT/IMPORT MDP
    def save(self, path):
        """
        Saves the MDP model and all the attributes needed into two files.

        .gpickle: MDP Model
        .pickle: Attributes
        """
        file_name = path + self.name + '_Graph.gpickle'
        print('---------------------------------')
        print('* SAVE MODEL')
        print('\t- MDP Graph:', file_name)
        nx.write_gpickle(self.pg, file_name)
        self._save_attributes(path)

    def load(self, path):
        """
        Loads the MDP model and all the attributes needed from two files.

        .gpickle: MDP Model
        .pickle: Attributes
        """

        file_name = path + self.name + '_Graph.gpickle'
        print('---------------------------------')
        print('* LOAD MODEL')
        print('\t- MDP Graph:', file_name)
        loaded_pg = nx.read_gpickle(file_name)
        assert type(loaded_pg) is type(self.pg), f'ERROR LOADING: ' \
                                                 f'MDP was saved as a {type(loaded_pg)}, not as nx.MultiDiGraph'
        self.pg = loaded_pg
        self._load_attributes(path)

    def _save_attributes(self, path):
        """
        Saves all interesting attributes into a pickle file
        """
        file_name = path + self.name + '_attributes.pickle'
        data = dict()

        # Frequencies
        data['Frequencies'] = dict({key: dict(d) for key, d in self.frequencies.items()})

        # Agent Metrics
        data['Agents Metrics'] = self.agent_metrics

        # MDP Metrics
        data['MDP Metrics'] = self.pg_metrics

        # Discretizer
        data['Discretizer'] = self.discretizer
        print(type(self.discretizer))

        print('\t- Attributes:', file_name)
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)

    def _load_attributes(self, path):
        """
        Loads all interesting attributes from a pickle file
        """
        file_name = path + self.name + '_attributes.pickle'
        print('\t- Attributes:', file_name)
        with open(file_name, 'rb') as file:
            data = pickle.load(file)

        # Frequencies
        self.frequencies = data['Frequencies']

        # Agent Metrics
        self.agent_metrics = data['Agents Metrics']

        # MDP Metrics
        self.pg_metrics = data['MDP Metrics']

        # Discretizer
        self.discretizer = data['Discretizer']