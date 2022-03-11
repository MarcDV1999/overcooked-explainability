# Extracts the Policy Graph of a given agent

from Code.Explainability.PolicyGraph import PolicyGraph

if __name__ == '__main__':

    ################## PARAMETERS ##################
    agent_id = 2
    ego_file = 'PantheonRL/models/ego{}'.format(agent_id)
    # FIXME: Igual aqui no hauriem de passar-li un alt, sino fer un load del mateix Ego. I aixi tenir dos Ego PPO
    alt_file = 'PantheonRL/models/alt{}'.format(agent_id)

    total_episodes = 100
    seeds = range(1,10,2)

    ################## COMPUTING PG ##################
    # Computes the PG of the given ego agent
    pg = PolicyGraph(ego_file, alt_file, total_episodes, seeds)



