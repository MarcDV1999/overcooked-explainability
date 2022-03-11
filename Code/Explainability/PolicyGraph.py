from argparse import Namespace

from Code.PantheonRL.trainer import generate_env, LAYOUT_LIST
from Code.PantheonRL.tester import generate_agent
from time import sleep
import numpy as np
from Code.PantheonRL.overcookedgym.human_aware_rl.overcooked_ai.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState


# Class that computes the Policy Graph of an ego agent
class PolicyGraph():

    ## ENVIRONMENT PARAMETERS
    ego = None
    alt = None
    env = None
    altenv = None

    def __init__(self, ego_file, alt_file, total_episodes, seeds):

        self.seeds = seeds

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

        ## POLICY GRAPH
        self.pg = None

        # Generates the environment, agents...
        self.generate_game()

        # Start to play the game
        self.play()

    ################## RUNNING AGENTS AND ENV ##################
    # Generates a new Game (Agents, Env, ...) using self.params
    def generate_game(self):
        # Creates the environment with the given configuration
        print(f"Arguments: {self.params}")
        self.env, self.altenv = generate_env(self.params)
        print(f"Environment: {self.env}; Partner env: {self.altenv}\n")

        # Creates the EGO Agent with the given configuration
        self.ego = generate_agent(self.env, self.params.ego, self.params.ego_config, self.params.ego_load)
        #print(f'Ego: {self.ego} - Policy: {self.ego.policy}\n')

        # Creates the ALT Agent with the given configuration and add partner
        self.alt = generate_agent(self.altenv, self.params.alt, self.params.alt_config, self.params.alt_load)
        self.env.add_partner_agent(self.alt)
        #print(f'Alt: {self.alt} - Policy: {self.alt.policy}\n')

        #print('Env Action Sample:', action_num_to_char(self.env.action_space.sample()), type(self.env))
        #print('Env OBS sample:', self.env.observation_space.sample(), type(self.env))
        #print('Env:', self.env.env.base_env, self.env, self.env.env.observation_space)

    # Play an Epoch in current env in order to feed the PG
    # An epoch it is formed by a number of episodes
    def play_epoch(self, num_episodes, render=False):

        rewards = []
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            reward = 0
            if render: self.env.render()
            while not done:
                # We get the action
                action = self.ego.get_action(obs, False)

                # Run step and save the obs
                obs, newreward, done, info = self.env.step(action)
                obs_state = info['Obs_str']
                obs_map = self.env.env.base_env.mdp.state_string(obs_state)

                # Update global reward
                reward += newreward

                # Update PG with current obs and action
                # obs: Observation featurized
                # obs_state: Normal Observation
                # obs_map: Map with the state of each pos
                self.update_pg(obs_state, action)

                if render:
                    self.env.render()
                    sleep(1 / 60)

            rewards.append(reward)

        self.env.close()
        print(f"Average Reward: {sum(rewards) / num_episodes}")
        print(f"Standard Deviation: {np.std(rewards)}")

    # Runs the (ego,alt) agent multiple epochs (Games) in different layouts and seeds
    # FIXME: El agent nomes te bons resultats en el entorn en el que ha estat entrenat
    def play(self):
        #layouts = LAYOUT_LIST
        layouts = ['simple']

        num_iters = len(layouts) * len(self.seeds)
        actual_iter = 1
        for layout in layouts:
            for seed in self.seeds:

                #start_time = time.time()
                self.params.env_config['layout_name'] = layout
                self.params.seed = seed

                # Generates the new game
                self.generate_game()

                # Play an epoch
                self.play_epoch(self.params.total_episodes, self.params.render)

                # Compute how much time we spent
                #end_time = time.time()
                print(f'Iteration Completed {actual_iter}/{num_iters} ({100 * actual_iter // num_iters}%)')
                print('---------------------------------')
                #print(f'Time Estimate: {(end_time-start_time)*(num_iters-actual_iter)}s')
                actual_iter += 1


    ################## POLICY GRAPH ##################
    # TODO: Updates the PG
    def update_pg(self, obs: OvercookedState, act):
        # Aqui haurem de discretitzar les observacions
        #print('Obs:', obs, 'Act:', act)

        # Position and Orientation of each Player
        (pos1, or1), (pos2, or2) = obs.players_pos_and_or

        # Unowned objects. The type is defaultdict{(obj_name: ObjState)}, example {'onion': [onion@(6, 1)]}
        unowned_objects = obs.unowned_objects_by_type
        #print(unowned_objects)

        # Owned objects. The type is defaultdict{(obj_name: ObjState)}, example {'onion': [onion@(6, 1)]}
        player_objects = obs.player_objects_by_type
        #print(player_objects)

        # Owned and Unowned objects
        all_objects_dic, all_objects_list = obs.all_objects_by_type, obs.all_objects_list
        #print(all_objects_dic, all_objects_list)

        # Current Order
        curr_order = obs.curr_order
        #print(curr_order)

        # Next Order
        next_order = obs.next_order
        #print(next_order)

        # Num orders remaining
        num_ord_rem = obs.num_orders_remaining
        #print(num_ord_rem)
