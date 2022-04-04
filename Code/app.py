# Extracts the Policy Graph of a given agent
import networkx as nx

from Code.Explainability.Discretizers.Discretizer_1 import Discretizer_1
from Code.Explainability.PolicyGraph import PolicyGraph
import sys

sys.path.append("../Code")

if __name__ == '__main__':

    ################## PARAMETERS ##################
    agent_id = '_simple'
    ego_file = 'PantheonRL/models/ego{}'.format(agent_id)
    # FIXME: Igual aqui no hauriem de passar-li un alt, sino fer un load del mateix Ego. I aixi tenir dos Ego PPO
    alt_file = 'PantheonRL/models/alt{}'.format(agent_id)

    # Num of games and which seeds
    total_episodes = 1
    #seeds = range(1,10,2)
    seeds = range(1, 40, 1)

    ################## COMPUTING PG ##################
    # Create the PG object
    discreitzer = Discretizer_1()
    pg = PolicyGraph(ego_file, alt_file, total_episodes, seeds, discreitzer)

    # Start feeding the PG
    #pg.feed(verbose=False)

    # Save the model
    # D1 esta entrenat amb 200, range(1, 40, 1)
    #pg.save('D2')

    pg.load('D1')


    ################## TESTING PG ##################
    # Runs the PG
    pg.test(total_episodes)

    ################## COMPARING AGENT AGAINST PG ##################
    #pg.compare()

    ################## SHOWING PG ##################

    pg.show(allow_recursion=True)



