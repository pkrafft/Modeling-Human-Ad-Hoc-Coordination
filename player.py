
import math
from utils import *

def play_with(model, player, companion_strategy, state, payoffs, maximize):
    """
    >>> import levelk
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> play_with(m, 0, lambda x: 0.74, (1,1,1), [4, 0, 3, 0], True)
    0
    >>> play_with(m, 0, lambda x: 0.76, (1,1,1), [5, 1, 4, 0], True)
    1
    >>> play_with(m, 0, lambda x: levelk.coordinate(m, 1, 1, x, False, None), (1,1,1), [10, 0, 9, 0], True)
    1
    >>> play_with(m, 0, lambda x: levelk.coordinate(m, 1, 2, x, False, None), (1,1,1), [10, 0, 9, 0], True)
    0
    >>> from asym_model import *
    >>> m = AsymmetricModel(0.1, 0.5)
    >>> play_with(m, 1, lambda x: levelk.coordinate(m, 0, 0, x, False, None), (1,0,1), [10, 0, 1, 0], True)
    0
    >>> play_with(m, 0, lambda x: levelk.coordinate(m, 1, 0, x, False, None), (1,1,0), [10, 0, 1, 0], True)
    1
    >>> play_with(m, 1, lambda x: levelk.coordinate(m, 0, 0, x, False, None), (1,1,1), [10, 0, 1, 0], True)
    1
    >>> from simple_model import *
    >>> m = SimpleModel(0.5, 0.95)
    >>> play_with(m, 0, lambda x: 1, (1,0,0), [4, 0, 3, 0], True)
    0
    >>> play_with(m, 0, lambda x: 1, (1,0,0), [4, 0, 3, 4], True)
    1
    """

    states = model.graphs[player][state]
    utils = {}
    for x in states:
        utils[x] = state_util(model, player, x, maximize, payoffs, companion_strategy)
    
    total_util = expected_util(model, player, state, utils)
    
    return action(maximize, total_util, payoffs)

def state_util(model, player, state, maximize, payoffs, companion_strategy, verbose = False):
    """
    >>> import levelk
    >>> from asym_model import *
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> state_util(m, 0, (1, 0, 1), True, [2,-1,1,0], lambda x: levelk.coordinate(m, 1, -1, x, True, [2,-1,1,0])) 
    0.20000000000000004
    >>> state_util(m, 1, (1, 0, 1), True, [2,-1,1,0], lambda x: levelk.coordinate(m, 0, -1, x, True, [2,-1,1,0])) 
    2.0
    >>> state_util(m, 1, (1, 0, 1), True, [2,-1,1,0], lambda x: levelk.coordinate(m, 0, 0, x, True, [2,-1,1,0])) 
    -1.0
    >>> state_util(m, 0, (1, 0, 1), False, [2,-1,1,0], lambda x: levelk.coordinate(m, 1, -1, x, False, None)) 
    0.10000000000000002
    >>> state_util(m, 1, (1, 0, 1), False, [2,-1,1,0], lambda x: levelk.coordinate(m, 0, -1, x, False, None)) 
    1.0
    >>> state_util(m, 1, (1, 0, 1), False, [2,-1,1,0], lambda x: levelk.coordinate(m, 0, 0, x, False, None)) 
    0.10000000000000001
    """
    
    p_dog = conditional(model, player, get_satisfying_states(model), state)
    
    p_other = companion_strategy(state)

    return get_util(maximize, p_dog, p_other, payoffs)

def expected_util(model, player, state, utils):
    """
    >>> from asym_model import *
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> expected_util(m, 0, (1, 1, 1), {(1, 1, 0):1, (1, 1, 1):2})
    1.25
    >>> expected_util(m, 1, (1, 1, 1), {(1, 1, 0):1, (1, 1, 1):2})
    2.0
    """
    
    norm = sum([model.probs[x] for x in model.graphs[player][state]])
    
    total_util = 0.0
    
    for x in model.graphs[player][state]:
        
        p_x = model.probs[x]/norm

        total_util += p_x * utils[x]
        
    return total_util


if __name__ == "__main__":
    import doctest
    doctest.testmod()
