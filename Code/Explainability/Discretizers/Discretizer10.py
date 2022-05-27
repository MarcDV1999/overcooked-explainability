from Explainability.Discretizers.Discretizer import Discretizer
from gym.wrappers.order_enforcing import OrderEnforcing
from Code.PantheonRL.overcookedgym.human_aware_rl.overcooked_ai.overcooked_ai_py.mdp.overcooked_mdp import \
    OvercookedState


class Discretizer10(Discretizer):
    """ Discretizer 10

    Predicates:
        - held
        - pot_state
    """

    NONE_VALUE = '*'
    PREDICATES = {
        'held':           [NONE_VALUE, 'O', 'T', 'D', 'S'],
        'pot_state' :     ['Of', 'Fi', 'Co', 'Wa'],
    }

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        super().__init__(env)


    def get_predicates(self, obs: OvercookedState):

        # Held Predicate
        held = self.get_held_predicate(obs, 0)

        # Pot state Predicate
        pot_state = self.get_pot_state_predicate(obs)

        # Build state
        # IMPORTANT!!! We have to add each predicate in the same order as the attribute self.PREDICATES !!!!
        predicate = {}
        predicate.update(held)
        predicate.update(pot_state)

        predicate_str = '-'.join(list(predicate.values()))
        return predicate, predicate_str
