
from utils import *

def common_p_belief(model, player, state, verbose = False):
    """
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> common_p_belief(m, 0, (1,1,0))
    0.95
    >>> common_p_belief(m, 1, (1,1,1))
    0.95
    >>> common_p_belief(m, 1, (0,1,1))
    0.0
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> common_p_belief(m, 0, (1,1,0,0,0))
    0.25000000000000017
    >>> from extended_thomas_model import *
    >>> m = ExtendedThomasModel(0.1, 0.25)
    >>> common_p_belief(m, 0, (1,1,0,0,0,0))
    0.25000000000000017
    >>> common_p_belief(m, 0, (1,1,0,1,0,0))
    0.10000000000000002
    >>> common_p_belief(m, 1, (1,1,1,0,1,1))
    0.25000000000000017
    >>> common_p_belief(m, 0, (1,1,1,1,1,0))
    0.49999999999999994
    """
    
    state_set = set(model.statespace)
    last_state_set = state_set
    
    while conditional(model, player, state_set, state) > 0:
        
        tmp_min_belief = get_min_belief(model, state_set)
        
        last_state_set = state_set
        state_set = super_p_evident(model, state_set, tmp_min_belief)

    min_belief = get_min_belief(model, last_state_set)

    if verbose:
        print last_state_set    
        
    return min_belief

def super_p_evident(model, state_set, p):
    """
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> super_p_evident(m, set([(1, 1, 1)]), 0.5)
    set([(1, 1, 1)])
    >>> super_p_evident(m, set([(1, 1, 1)]), 0.95)    
    set([])
    >>> super_p_evident(m, set([(1, 1, 0)]), 0.5)
    set([])
    >>> super_p_evident(m, set([(1, 1, 1), (1, 1, 0)]), 0.5)
    set([(1, 1, 1)])
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> super_p_evident(m, set([(1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]), 0)
    set([(1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 0)])
    >>> super_p_evident(m, set([(1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]), 0.26)
    set([(1, 1, 1, 1, 1), (1, 1, 1, 1, 0)])
    >>> super_p_evident(m, set([(1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]), 0.5)
    set([(1, 1, 1, 1, 1)])
    >>> super_p_evident(m, set([(1, 1, 1, 1, 1)]), 1)
    set([])
    """
    
    event = get_satisfying_states(model)
    
    changed = True
    while changed:
        changed = False
        for i in range(model.n_players):
            for x in state_set:
                if conditional(model, i, state_set, x) <= p or conditional(model, i, event, x) <= p:
                    state_set = state_set.difference(set([x]))
                    changed = True
    
    return state_set

def get_min_belief(model, state_set):
    """
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> get_min_belief(m, set(m.statespace))
    0.0
    >>> get_min_belief(m, set([(1, 1, 1)]))
    0.95
    >>> get_min_belief(m, set([(1, 1, 0)]))
    0.05000000000000007
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> get_min_belief(m, set([(1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]))
    0.25000000000000017
    >>> get_min_belief(m, set([(1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]))
    0.49999999999999994
    >>> get_min_belief(m, set([(1, 1, 1, 1, 1)]))
    1.0
    """
    
    event = get_satisfying_states(model)
    
    p = 1.0

    for i in range(model.n_players):
        
        for x in state_set:
            
            this_p = min(conditional(model, i, state_set, x), conditional(model, i, event, x))
            
            if this_p < p:
                p = this_p
    
    return p

if __name__ == "__main__":
    import doctest
    doctest.testmod()
