import time
from argparse import Namespace

from PantheonRL.trainer import generate_env, LAYOUT_LIST
from PantheonRL.tester import generate_agent
from time import sleep
import numpy as np

def action_num_to_char(action_num):
    if action_num == 0:
        return "↑"
    elif action_num == 1:
        return '↓'
    elif action_num == 2:
        return '→'
    elif action_num == 3:
        return '←'
    elif action_num == 4:
        return 'stay'
    elif action_num == 5:
        return 'interact'

# Class that computes the Policy Graph of an ego agent
class Policy_Graph():

    def __init__(self, ego_file, alt_file, total_episodes, seeds):

        ## ENVIRONMENT PARAMETERS
        self.ego = None
        self.alt = None
        self.env = None
        self.altenv = None
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
            record='traj.txt',
            render=False,
            seed=None,
            total_episodes=total_episodes)

        ## POLICY GRAPH
        self.pg = None

        # Generates the environment
        self.generate_game()

        # Start to play the game
        self.play()

    ################## RUNNING EGO AGENT ##################
    # Generates a new Game
    # Generates new agents and env with the current 'params'
    def generate_game(self):
        # Creates the environment with the given configuration
        #print(f"Arguments: {self.params}")
        self.env, self.altenv = generate_env(self.params)
        #print(f"Environment: {self.env}; Partner env: {self.altenv}\n")

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
    def play_epoch(self, num_episodes, render=False):

        rewards = []
        for game in range(num_episodes):
            obs = self.env.reset()
            done = False
            reward = 0
            if render: self.env.render()
            while not done:
                # We get the action
                action = self.ego.get_action(obs, False)
                # Run step
                obs, newreward, done, info = self.env.step(action)
                state_str = info['Obs_str']
                #print('Obs String:', state_str, type(state_str))
                #print('Obs String:', obs, type(obs))
                #print(self.env.env.base_env.mdp.state_string(state_str))

                # Update global reward
                reward += newreward

                # Update PG with current obs and action
                self.update_pg(obs, action)

                if render:
                    self.env.render()
                    sleep(1 / 60)

            rewards.append(reward)

        self.env.close()
        print(f"Average Reward: {sum(rewards) / num_episodes}")
        print(f"Standard Deviation: {np.std(rewards)}")

    # Runs the ego agent multiple epochs in different layouts and seeds
    def play(self):
        num_iters = len(LAYOUT_LIST) + len(self.seeds)
        actual_iter = 1
        for layout in LAYOUT_LIST:
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
    # Updates the PG
    def update_pg(self, obs, act):
        # Aqui haurem de discretitzar les observacions
        pass





if __name__ == '__main__':

    ################## PARAMETERS ##################
    ego_file = 'PantheonRL/models/ego1'
    alt_file = 'PantheonRL/models/alt1'
    total_episodes = 5
    seeds = range(1,10,2)

    ################## COMPUTING PG ##################
    # Computes the PG of the given ego agent
    pg = Policy_Graph(ego_file, alt_file, total_episodes, seeds)



