from PantheonRL.trainer import generate_env, LAYOUT_LIST
from PantheonRL.tester import generate_agent, run_test
from arguments_utils import input_check, get_arguments
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


class Policy_Graph():
    def __init__(self, agent):
        self.pg = None
        self.agent = agent

    def feed(self, args, seed):
        # Creates the environment with the given configuration
        print(f"Arguments: {args}")
        env, altenv = generate_env(args)
        print(f"Environment: {env}; Partner env: {altenv}\n")

        # Creates the EGO Agent with the given configuration
        ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
        print(f'Ego: {ego} - Policy: {ego.policy}\n')

        # Creates the ALT Agent with the given configuration and add partner
        alt = generate_agent(altenv, args.alt, args.alt_config, args.alt_load)
        env.add_partner_agent(alt)
        print(f'Alt: {alt} - Policy: {alt.policy}\n')

        print('Env Action Sample:', action_num_to_char(env.action_space.sample()), type(env))
        print('Env OBS sample:', env.observation_space.sample(), type(env))
        print('Env:', env.env.base_env, env, env.env.observation_space)

        # Start to play the game
        self.play(ego, env, args.total_episodes, args.render)

    def play(self, ego, env, num_episodes, render=False):
        rewards = []
        for game in range(num_episodes):
            obs = env.reset()
            done = False
            reward = 0
            if render: env.render()
            while not done:
                # We get the action
                action = ego.get_action(obs, False)
                # Run step
                obs, newreward, done, info = env.step(action)
                state_str = info['Obs_str']
                print('Obs String:', state_str, type(state_str))
                #print('Obs String:', obs, type(obs))
                print(env.env.base_env.mdp.state_string(state_str))

                # Update global reward
                reward += newreward

                if render:
                    env.render()
                    sleep(1 / 60)

            rewards.append(reward)

        env.close()
        print(f"Average Reward: {sum(rewards) / num_episodes}")
        print(f"Standard Deviation: {np.std(rewards)}")





if __name__ == '__main__':
    seeds = 2
    agent = None
    args = get_arguments()
    pg = Policy_Graph(agent)


    for layout in LAYOUT_LIST:
        for s in range(1, seeds):
            pg.feed(args, s)