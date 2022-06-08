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


def run_all_experiments(agent_ids, layouts,
                        pg_algorithms,
                        discretizers,
                        uuid,
                        batch_size=25,
                        test_seeds=15, test_episodes=4,
                        train_seeds=500, train_episodes=3,
                        description=""):
    for i in range(len(layouts)):
        for discretizer in discretizers:
            for pg_algorithm in pg_algorithms:
                experiment = Experiment(id=uuid, rl_agent_id=agent_ids[i],
                                        discretizer=discretizer,
                                        pg_algorithm=pg_algorithm,
                                        layout=layouts[i],
                                        description=description)

                experiment.feed_and_test(batch_size=batch_size,
                                         test_seeds=test_seeds, test_episodes=test_episodes,
                                         train_seeds=train_seeds, train_episodes=train_episodes)


if __name__ == '__main__':
    agent_ids =   ['_simple1M', '_unident_s1M', '_random0_1M', '_random1_1M', '_random3_1M']
    layouts =     ['simple', 'unident_s', 'random0', 'random1', 'random3']

    agent_ids = ['_random3_1M']
    layouts = ['random3']

    # Run all the experiments at once
    run_all_experiments(uuid="New_Experiment_500_25",
                        agent_ids=agent_ids[::-1],
                        layouts=layouts[::-1],
                        batch_size=25,
                        test_seeds=15, test_episodes=4,
                        train_seeds=500, train_episodes=3,
                        discretizers=[Discretizer11, Discretizer12, Discretizer13][::-1],
                        pg_algorithms=[PartialPolicyGraph, CompletePolicyGraph],
                        description=""
                        )

