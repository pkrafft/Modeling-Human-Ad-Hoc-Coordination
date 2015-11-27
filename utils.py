
import numpy as np
import math

def state_probs(model, player, state):
    """
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> state_probs(m, 0, (1,1,0))
    ([(1, 1, 0), (1, 1, 1)], [0.050000000000000072, 0.94999999999999996])
    """

    possible = model.graphs[player][state]
    
    norm = sum([model.probs[x] for x in possible])
    p = [model.probs[x]/norm for x in possible]

    return possible, p

def conditional(model, player, state_set, x):
    """
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> conditional(m, 0, set([(1, 1, 1)]), (1,1,0))
    0.95
    >>> conditional(m, 0, set([(1, 1, 1)]), (1,0,0))
    0.0
    >>> conditional(m, 0, get_satisfying_states(m), (1,0,0))
    0.10000000000000002
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> conditional(m, 0, get_satisfying_states(m), (1,1,1,0,0))
    1.0
    >>> conditional(m, 0, set([(1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]), (1,1,1,0,0))
    0.25000000000000017
    >>> conditional(m, 0, set([(1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]), (1,1,1,1,1))
    1.0
    >>> from extended_thomas_model import *
    >>> m = ExtendedThomasModel(0.1, 0.25, 2)
    >>> conditional(m, 0, get_satisfying_states(m), (1,1,0,0,0,0,0,0,0,0,0))
    1.0
    """
    
    possible_states = set(model.graphs[player][x])
    state_set = state_set.intersection(possible_states)
    norm = sum([model.probs[x] for x in possible_states])
    if len(state_set) > 0:
        p = sum([model.probs[x] for x in state_set])
        p = math.exp( math.log(p) - math.log(norm) )
    else:
        p = 0.0
        assert norm > 0
    
    return p
            
def get_satisfying_states(model):
    """
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> get_satisfying_states(m)
    set([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)])
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> get_satisfying_states(m)
    set([(1, 1, 0, 1, 0), (1, 1, 1, 0, 0), (1, 1, 1, 1, 0), (1, 0, 1, 0, 0), (1, 0, 1, 1, 0), (1, 1, 0, 1, 1), (1, 0, 1, 1, 1), (1, 0, 1, 0, 1), (1, 1, 1, 1, 1), (1, 0, 0, 1, 1), (1, 0, 0, 0, 0), (1, 1, 1, 0, 1), (1, 1, 0, 0, 0), (1, 0, 0, 1, 0), (1, 1, 0, 0, 1), (1, 0, 0, 0, 1)])
    >>> m = ThomasModel(0.1, 0.25, 2)
    >>> (1, 0, 1, 0, 0, 1, 0, 0, 0) in get_satisfying_states(m)
    True
    """
    satisfying = []
    for x in model.statespace:
        if x[0] == 1:
            satisfying += [x]
    return set(satisfying)

def log_bernoulli(x, p):
    """
    >>> log_bernoulli(1, np.exp(-1))
    -1.0
    >>> log_bernoulli(0, 1 - np.exp(-2))
    -2.0
    """
    if x == 1:
        return np.log(p)
    else:
        return np.log(1 - p)

def print_partition(partitions):
    for i in range(len(partitions)):
        print 'player', i
        for x in partitions[i]:
            print(str(x) + '   ' + str([partitions[i][x][j] for j in range(len(partitions[i][x]))]))

def get_util(maximize, p_dog, p_other, payoffs):
    """
    >>> get_util(True, 1.0, 1.0, [1,2,3,4])
    1.0
    >>> get_util(True, 1.0, 0.5, [1,2,3,4])
    1.5
    >>> get_util(True, 0.5, 0.5, [1,2,3,4])
    2.25
    >>> get_util(True, 0.0, 1.0, [1,2,3,4])
    4.0
    >>> get_util(True, 1.0, 0.0, [1,2,3,4])
    2.0
    >>> get_util(False, 1.0, 1.0, [1,2,3,4])
    1.0
    >>> get_util(False, 1.0, 0.5, [1,2,3,4])
    0.5
    >>> get_util(False, 0.5, 0.5, [1,2,3,4])
    0.25
    >>> get_util(False, 0.0, 1.0, [1,2,3,4])
    0.0
    >>> get_util(False, 1.0, 0.0, [1,2,3,4])
    0.0
    """
    
    if maximize:
        util = 0
        util += p_dog * p_other * payoffs[0]
        util += (1 - p_dog) * p_other * payoffs[3]
        util += (1 - p_other) * payoffs[1]
    else:
        util = p_dog * p_other

    return util 

def action(maximize, total_util, payoffs):
    """
    >>> action(True, 0.9, [1.1, 0, 1, 0.4])
    0
    >>> action(True, 1.1, [1.1, 0, 1, 0.4])
    1
    >>> action(False, 0.9, [1.1, 0, 1, 0.4])
    0.9
    """
    
    if maximize:
        return int(total_util > payoffs[2]) 
    else:
        return total_util

def optimal_threshold(payoffs):
    """
    >>> optimal_threshold([2,0,1,0.5])
    0.5
    >>> optimal_threshold([1.1,0,1,0.4])
    0.9090909090909091
    """
    return (payoffs[2] - payoffs[1]) / float(payoffs[0] - payoffs[1])

if __name__ == "__main__":
    import doctest
    doctest.testmod()

