import sys

sys.path.append("./Code")
sys.path.append("./Code/PantheonRL/overcookedgym/human_aware_rl/overcooked_ai")

from Code.Explainability.Discretizers.Discretizer10 import Discretizer10
from Code.Explainability.Discretizers.Discretizer11 import Discretizer11
from Code.Explainability.Discretizers.Discretizer12 import Discretizer12
from Code.Explainability.Discretizers.Discretizer13 import Discretizer13
from Code.Explainability.Discretizers.Discretizer14 import Discretizer14
from Code.Explainability.PartialPolicyGraph import PartialPolicyGraph
from Code.Explainability.CompletePolicyGraph import CompletePolicyGraph
from Code.Experiment import Experiment

if __name__ == '__main__':
    # agent_ids =   ['_simple1M', '_unident_s1M', '_random0_1M', '_random1_1M', '_random3_1M']
    # layouts =     ['simple', 'unident_s', 'random0', 'random1', 'random3']

    # Run one experiment only
    experiment = Experiment(id='2_Experiment_500_25', rl_agent_id='_simple1M',
                            discretizer=Discretizer14,
                            pg_algorithm=CompletePolicyGraph,
                            layout='simple'
                            )

    experiment.run(train=False,
                   test=False,
                   subgraph=[4,5])
    # Subgraph: Find a node in the MDP such as its total edges is between [min, max]. Once we found the node, save its
    # subgraph
