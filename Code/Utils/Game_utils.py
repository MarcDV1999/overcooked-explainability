from Code.PantheonRL.tester import generate_agent
from Code.PantheonRL.trainer import generate_env


def generate_game(params, verbose=False):
    """
    Generates a new Game (Agents, Env, ...) using params.
    """

    # Creates the environment with the given configuration
    env, altenv = generate_env(params)

    # Creates the EGO Agent with the given configuration
    ego = generate_agent(env, params.ego, params.ego_config, params.ego_load)

    # Creates the ALT Agent with the given configuration and add partner
    alt = generate_agent(altenv, params.alt, params.alt_config, params.alt_load)
    env.add_partner_agent(alt)

    if verbose:
        print(f"Arguments: {params}")
        print(f"Environment: {env}; Partner env: {altenv}\n")
        print(f'Ego: {ego} - Policy: {ego.policy}\n')
        print(f'Alt: {alt} - Policy: {alt.policy}\n')

    return env, altenv, ego, alt
