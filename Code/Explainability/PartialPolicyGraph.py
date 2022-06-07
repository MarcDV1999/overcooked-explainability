from Code.Utils.utils import action_num_to_char, get_assigned_color
import networkx as nx
import numpy as np

from Explainability.PolicyGraph import PolicyGraph


class PartialPolicyGraph(PolicyGraph):
    """ Record only the most probable action for each state

    This algorithm builds a directed graph. For each state, it takes the most probable action.
    Therefore, we do not add all the agent interactions to our graph, we only pay attention to the interactions
    that belong to the most used action by the agent. This means that for each node, we only have one possible action
    (the most probable in this case). We think this could be an interesting approach since we are building a
    deterministic agent. Probably this algorithm can work well in simpler layouts where decision-making is easier.
    """

    MDP_MODELS_FOLDER = 'Code/MDP_Models/'

    def __init__(self, ego_file, alt_file, discretizer, name, layout):
        """
        Constructor.

        params (Namespace): Parameters of the environment.
        discretizer (Discretizer): Object that converts OvercookedStates into discrete predicates.
        pg: Graph
        """
        super().__init__(ego_file, alt_file, discretizer, name, layout)
        self.pg = nx.DiGraph(name='MDP')

    def build_mdp(self, verbose=False):
        """
        Takes frequencies and builds the Markov Decision Graph.

        For each (node) we take the most frequent action.
        Then we add to the graph all the edges (node, best_action, next_node)

        In this algorithm, each node only saves the best action for this reason, each node only has one possible action
        """
        self.pg = nx.DiGraph(name='MDP')
        # For each state, we take into account the most probable action
        for state, actions in self.frequencies.items():
            action_keys = np.array(list(actions.keys()))

            freq_actions = np.array([sum(list(actions[a].values())) for a in action_keys], dtype=np.int)
            freq_actions_sum = np.sum(freq_actions)

            prob_actions = freq_actions / freq_actions_sum
            prob_actions_sum = np.sum(prob_actions)

            action_more_prob = action_keys[np.argmax(prob_actions)]

            next_states = actions[action_more_prob]

            freq_next = np.array([freq for _, freq in list(next_states.items())], dtype=np.int)
            freq_next_sum = np.sum(freq_next)

            prob_next_s = freq_next / freq_next_sum
            prob_next_s_sum = np.sum(prob_next_s)

            if verbose:
                print('We are in state:', state)
                print('\tWe chose actions:\t\t', action_keys)
                print('\t\tFrequencies:\t\t\t', freq_actions, 'Total:', freq_actions_sum)
                print('\t\tProbs:\t\t\t\t\t', prob_actions, 'Total:', prob_actions_sum)
                print('\t\tMost probable action:', action_more_prob, 'amb prob:', prob_actions[np.argmax(prob_actions)])
                print('\t\t\tNext states action', action_more_prob, ':', list(next_states.items()))
                print('\t\t\tFrequencies', freq_next, 'Total:', freq_next_sum)
                print('\t\t\tProbabilities', prob_next_s, 'Total:', prob_next_s_sum)

            # Add new_edges to the Graph
            new_edges = list(zip([state] * len(list(next_states.keys())), list(next_states.keys()), prob_next_s))

            for _, next_state, prob in new_edges:

                if verbose:
                    print('Append the edges:', new_edges, 'amb la accio:', action_num_to_char(action_more_prob))
                    print('---------')
                # label: In order to show the edge label with pyvis
                # width: In order to show the edge width with pyvis
                color = get_assigned_color(action_more_prob)
                self.update_edge(state, next_state, int(action_more_prob), prob, color)

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

        Here we used the first one, but other options could work.

        :param predicate: Existent or non-existent state
        :param verbose: Prints additional information
        :return: Action
        """

        # Predicate does not exist
        if predicate not in self.pg.nodes:
            self.pg_metrics['new_state'] += 1

        nearest_predicate = self.get_nearest_predicate(predicate, verbose=verbose)
        possible_actions = self.get_possible_actions(nearest_predicate)

        # Possible actions always will have 1 element since  for each state we only save the best action
        return possible_actions[0][0]

    def update_edge(self, u, v, a, w, color):
        # Edge does not exist
        self.pg.add_edge(u, v, action=a, weight=w, color=color)
        return True
