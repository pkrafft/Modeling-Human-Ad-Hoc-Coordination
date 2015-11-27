
import random
import math

from utils import *
from player import *
    
def coordinate(model, p, k, state, maximize, payoffs, verbose = False):
    """
    >>> from asym_model import *
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> coordinate(m, 1, 0, (1, 0, 1), True, [2,0,1,0])
    1
    >>> coordinate(m, 1, 1, (1, 0, 1), True, [2,0,1,0])
    0
    >>> coordinate(m, 0, 0, (1, 0, 1), True, [2,0,1,0])
    0
    >>> coordinate(m, 0, 1, (1, 1, 1), True, [2,0,1,0])
    0
    >>> m = AsymmetricModel(0.1, 0.75)
    >>> coordinate(m, 0, 1, (1, 1, 1), True, [2,0,1,0], True)
    1 (1, 1, 1)
    0 (1, 1, 0)
    0 (1, 1, 1)
    1
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> coordinate(m, 1, 0, (1, 0, 1), False, [2,0,1,0])
    1.0
    >>> coordinate(m, 1, 1, (1, 0, 1), False, [2,0,1,0])
    0.10000000000000001
    >>> coordinate(m, 0, 0, (1, 0, 1), False, [2,0,1,0])
    0.10000000000000001
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> coordinate(m, 0, 1, (1, 1, 1), False, [2,0,1,0])
    0.32499999999999996
    >>> c1 = coordinate(m, 0, 1, (1, 1, 1), False, [2,0,1,0])
    >>> m = AsymmetricModel(0.1, 0.75)
    >>> c2 = coordinate(m, 0, 1, (1, 1, 1), False, [2,0,1,0])
    >>> c2 > c1
    True
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> c1 = coordinate(m, 1, 2, (1, 1, 1), False, [2,0,1,0])
    >>> m = AsymmetricModel(0.1, 0.75)
    >>> c2 = coordinate(m, 1, 2, (1, 1, 1), False, [2,0,1,0])
    >>> c2 > c1
    True
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> c1 = coordinate(m, 0, 1, (1, 1, 1), False, [2,0,1,0])
    >>> m = AsymmetricModel(0.2, 0.25)
    >>> c2 = coordinate(m, 0, 1, (1, 1, 1), False, [2,0,1,0])
    >>> c2 > c1
    True
    """

    if k >= 0:
        
        if verbose:
            print k, state
        
        other = lambda x: coordinate(model, 1 - p, k - 1, x, maximize, payoffs, verbose)
        
        return play_with(model, p, other, state, payoffs, maximize)
    
    else: # level 0 player assumes the other player takes jointly optimal action
        
        return 1.0 # other player coordinates if needed, otherwise doesn't matter


if __name__ == "__main__":
    import doctest
    doctest.testmod()
