from Code.PantheonRL.tester import generate_agent
from Code.PantheonRL.trainer import generate_env


# Generates a new Game (Agents, Env, ...) using self.params
def generate_game(params):
    # Creates the environment with the given configuration
    print(f"Arguments: {params}")
    env, altenv = generate_env(params)
    print(f"Environment: {env}; Partner env: {altenv}\n")

    # Creates the EGO Agent with the given configuration
    ego = generate_agent(env, params.ego, params.ego_config, params.ego_load)
    # print(f'Ego: {self.ego} - Policy: {self.ego.policy}\n')

    # Creates the ALT Agent with the given configuration and add partner
    alt = generate_agent(altenv, params.alt, params.alt_config, params.alt_load)
    env.add_partner_agent(alt)
    # print(f'Alt: {self.alt} - Policy: {self.alt.policy}\n')

    # print('Env Action Sample:', action_num_to_char(self.env.action_space.sample()), type(self.env))
    # print('Env OBS sample:', self.env.observation_space.sample(), type(self.env))
    # print('Env:', self.env.env.base_env, self.env, self.env.env.observation_space)
    return env, altenv, ego, alt