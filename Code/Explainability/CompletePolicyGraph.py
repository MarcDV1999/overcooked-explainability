from Code.Utils.utils import get_assigned_color, Actions
import networkx as nx
import numpy as np

from Explainability.PolicyGraph import PolicyGraph


class CompletePolicyGraph(PolicyGraph):
    """ Record all the transitions.

    This algorithm builds a multi-directed graph. For each state, it takes all the agent interactions.
    Therefore, we add them all to our graph. This means that for each node, we have multiple possible actions.
    We think this could be an interesting approach since we are maintaining stochasticity. Probably this algorithm
    can work better in more complex layouts where decision-making is more relative.
    """

    MDP_MODELS_FOLDER = 'Code/MDP_Models/'

    def __init__(self, ego_file, alt_file, discretizer, name, layout):
        """
        Constructor.

        params (Namespace): Parameters of the environment.
        discretizer (Discretizer): Object that converts OvercookedStates into discrete predicates.
        pg: Graph
        frequencies: Saves how many times (state, action, next state) is reached.
        """
        super().__init__(ego_file, alt_file, discretizer, name, layout)
        self.pg = nx.MultiDiGraph(name='MDP')

    # MDP Generation
    def build_mdp(self, verbose=False):
        """
        Takes frequencies and builds the Markov Decision Graph.

        In this case, we add al the possible edges.
        Then we add to the graph all the edges (node, action, next_node)

        In this algorithm, each node can have multiple edges with different actions
        """
        for state, actions in self.frequencies.items():
            # Times per action
            sum_partial = {action: sum(list(next_states.values())) for action, next_states in
                           self.frequencies[state].items()}
            # Total times
            sum_total = sum(list(sum_partial.values()))

            # Percentage per action
            prob_sub_partial = self.frequencies[state].copy()
            prob_sub_partial = {action: {next_s: freq / sum_total for next_s, freq in prob_sub_partial[action].items()}
                                for action, _ in prob_sub_partial.items()}

            new_edges = []
            for action, next_states in prob_sub_partial.items():
                new_edge = list(zip([state] * len(next_states.values()),
                                    [action] * len(next_states.values()),
                                    list(next_states.keys()),
                                    list(next_states.values()),
                                    )
                                )
                new_edges.append(new_edge)

            for edge_list_by_actions in new_edges:
                for u, a, v, p in edge_list_by_actions:
                    # label: In order to show the edge label with pyvis
                    # width: In order to show the edge width with pyvis
                    color = get_assigned_color(a)
                    self.update_edge(u, v, int(a), p, color)

            self.normalize_node_weights(state)

        nx.set_node_attributes(self.pg,
                               {node: self.discretizer.get_predicate_label(node)
                                for node, freq in self.frequencies.items()},
                               name='label')

    def select_action_using_mdp(self, predicate, verbose):
        """
        Given a predicate, goes to the MDP and selects the corresponding action.
        Since each node can have multiple possible actions, we can face this problem in different ways:

        - Take the most probable action
        - Take one action using its probability distribution

        Here we used the second one, but other options could work

        :param predicate: Existent or non-existent state
        :param verbose: Prints additional information
        :return: Action
        """
        # Predicate does not exist
        if predicate not in self.pg.nodes:
            self.pg_metrics['new_state'] += 1

        nearest_predicate = self.get_nearest_predicate(predicate, verbose=verbose)
        possible_actions = self.get_possible_actions(nearest_predicate)

        # Probability distribution
        p = [data[1] for data in possible_actions]
        a = [data[0].value for data in possible_actions]

        sum_prob = sum(p)
        assert (sum_prob > 0.999 and sum_prob < 1.001), f"{sum(p)} - {p}"

        # Take one action with a given Probability distribution
        p = np.array(p)
        p /= p.sum()
        rand_action = np.random.choice(a, p=p)
        rand_action = Actions(rand_action)

        return rand_action

    def update_edge(self, u, v, a, w, color):
        # Edges
        edges = self.pg.get_edge_data(u, v)
        if edges is not None:
            edges = list(edges.values())
            for edge in edges:
                # Exists the edge
                if edge['action'] == a:
                    # Compute the new weight
                    # edge['weight'] = (alpha * w) + ((1 - alpha) * edge['weight'])
                    edge['weight'] = w
                    return True

        # Edge does not exist
        self.pg.add_edge(u, v, action=a, weight=w, color=color)
        return True
