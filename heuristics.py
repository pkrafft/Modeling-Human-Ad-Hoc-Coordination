
import random
import math
from utils import *

# the player knows 
def alone(model, player, state):
    """
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> alone(m, 0, (1,1,1,0,0))
    1
    >>> alone(m, 1, (1,1,0,0,0))
    0
    >>> alone(m, 0, (1,1,1,1,0))
    1
    """

    prob_dog = conditional(model, player, get_satisfying_states(model), state)
    return int(prob_dog > 1 - 1e-12)

# the player knows both players know
def together(model, player, state):
    """
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> together(m, 0, (1,1,1,0,0))
    0
    >>> together(m, 1, (1,1,1,0,0))
    1
    >>> together(m, 0, (1,1,1,0,1))
    0
    >>> together(m, 0, (1,1,1,1,0))
    1
    >>> together(m, 1, (1,1,1,1,0))
    1
    >>> from speaker_model import *
    >>> m = SpeakerModel(0.1, 0.25)
    >>> together(m, 0, (1,1))
    1
    >>> together(m, 1, (1,1))
    1
    >>> together(m, 0, (1,0))
    0
    """
    
    prob_dog = conditional(model, player, get_satisfying_states(model), state)
    if prob_dog < 1 - 1e-12:
        return 0

    for x in model.graphs[player][state]:
        
        prob_dog = conditional(model, 1 - player, get_satisfying_states(model), x)
        if prob_dog < 1 - 1e-12:
            return 0

    return 1

if __name__ == "__main__":
    import doctest
    doctest.testmod()
