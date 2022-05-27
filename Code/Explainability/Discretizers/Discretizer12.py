from Explainability.Discretizers.Discretizer import Discretizer
from gym.wrappers.order_enforcing import OrderEnforcing
from Code.PantheonRL.overcookedgym.human_aware_rl.overcooked_ai.overcooked_ai_py.mdp.overcooked_mdp import \
    OvercookedState


class Discretizer12(Discretizer):
    """Discretizer

    Predicates:
        - held
        - held_partner
        - pot_state
        - predicate_pos
    """

    NONE_VALUE = '*'
    PREDICATES = {
        'held':           [NONE_VALUE, 'O', 'T', 'D', 'S'],
        'held_partner':   [NONE_VALUE, 'O', 'T', 'D', 'S'],
        'pot_state' :     ['Of', 'Fi', 'Co', 'Wa'],
        'onion_pos' :     ['S', 'R', 'L', 'T', 'B', 'I'],
        'tomato_pos' :    ['S', 'R', 'L', 'T', 'B', 'I'],
        'dish_pos':       ['S', 'R', 'L', 'T', 'B', 'I'],
        'pot_pos' :       ['S', 'R', 'L', 'T', 'B', 'I'],
        'service_pos' :   ['S', 'R', 'L', 'T', 'B', 'I'],
        'soup_pos' :      ['S', 'R', 'L', 'T', 'B', 'I'],
    }

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        super().__init__(env)


    def get_predicates(self, obs: OvercookedState):

        # Held Predicate
        held = self.get_held_predicate(obs, 0)

        # Held Partner Predicate
        held_partner = self.get_held_predicate(obs, 1)

        # Pot state Predicate
        pot_state = self.get_pot_state_predicate(obs)

        # Temporary sources (coordination)
        temporary_sources = self.get_temporary_sources(obs)

        # Onion pos predicate
        onion_pos = self.get_onion_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0)

        # Tomato pos predicate
        tomato_pos = self.get_tomato_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0)

        # Soup pos predicate
        soup_pos = self.get_soup_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0)

        # Dish pos predicate
        dish_pos = self.get_dish_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0)

        # Service pos predicate
        service_pos = self.get_service_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0)

        # Pot pos predicate
        pot_pos = self.get_pot_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0)

        # Build state
        # IMPORTANT!!! We have to add each predicate in the same order as the attribute self.PREDICATES !!!!
        predicate = {}
        predicate.update(held)
        predicate.update(held_partner)
        predicate.update(pot_state)
        predicate.update(onion_pos)
        predicate.update(tomato_pos)
        predicate.update(soup_pos)
        predicate.update(dish_pos)
        predicate.update(service_pos)
        predicate.update(pot_pos)

        predicate_str = '-'.join(list(predicate.values()))
        return predicate, predicate_str
