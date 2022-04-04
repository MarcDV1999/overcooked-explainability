from Code.PantheonRL.overcookedgym.human_aware_rl.overcooked_ai.overcooked_ai_py.mdp.overcooked_mdp import \
    OvercookedState, PlayerState
from functools import reduce
import operator


from itertools import product

############### PREDICATES

### Oven: State of the oven
#       - Finished: The soup is ready to take
#       - Cooking:  The onions are being cooked
#       - Off:      The oven is off, without onions
#       - Waiting:  The oven has some onions but not 3.

### Held: What is holding the player
#       - None
#       - Onion
#       - Soup
#       - Dish


class Discretizer_1:
    def __init__(self):
        # Possible predicates
        self.predicate_space = {
            'held': [None, 'onion', 'dish', 'soup'],
            'oven': ['off', 'finished', 'cooking', 'waiting']
        }

        self.num_states = reduce(operator.mul, [len(l) for l in self.predicate_space.values()])
        self.initial_state = 'None-off'

    # Method that discretizes the OvercookedState obs
    def get_predicates(self, obs: OvercookedState):
        # If invalid state, return default state (initial)
        if obs is None: return self.initial_state

        player1: PlayerState = obs.players[0]

        # Held Predicate
        if player1.has_object():
            held = player1.get_object().name
        else:
            held = None

        # Oven Predicate
        unowned_objects = obs.unowned_objects_by_type
        if 'soup' in unowned_objects:
            time = unowned_objects['soup'][0].state[2]
            num_onions = unowned_objects['soup'][0].state[1]
            if num_onions == 3 and time == 20:
                oven = 'finished'
            elif num_onions == 3:
                oven = 'cooking'
            else:
                oven = 'waiting'
        else:
            oven = 'off'

        # Build state
        predicates = str(held) + '-' + str(oven)
        return predicates

    def get_possible_states(self):
        states = []
        colors = []
        index = 0
        for element in product(*self.predicate_space.values()):
            states.append(str(element[0]) + '-' + str(element[1]))
            colors.append('red')
            index += 1
        return states, colors

    def get_possible_actions(self):
        return range(6)
