
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
    
    evident_event = set(model.statespace)
    last_evident_event = evident_event
    
    while conditional(model, player, evident_event, state) > 0:
        
        tmp_min_belief = evidence_level(model, evident_event)
        
        last_evident_event = evident_event
        evident_event = super_p_evident(model, evident_event, tmp_min_belief)

    min_belief = evidence_level(model, last_evident_event)

    if verbose:
        print last_evident_event    
        
    return min_belief

def super_p_evident(model, evident_event, p):
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
    
    dog_day_event = get_satisfying_states(model)
    
    changed = True
    while changed:
        changed = False
        for x in evident_event:
            if get_min_belief(model, evident_event, [x]) <= p:
                evident_event = evident_event.difference(set([x]))
                changed = True
    
    return evident_event

def evidence_level(model, evident_event):
    """
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> evidence_level(m, set(m.statespace))
    0.0
    >>> evidence_level(m, set([(1, 1, 1)]))
    0.95
    >>> evidence_level(m, set([(1, 1, 0)]))
    0.05000000000000007
    >>> from thomas_model import *
    >>> m = ThomasModel(0.1, 0.25)
    >>> evidence_level(m, set([(1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]))
    0.25000000000000017
    >>> evidence_level(m, set([(1, 1, 1, 1, 1), (1, 1, 1, 1, 0)]))
    0.49999999999999994
    >>> evidence_level(m, set([(1, 1, 1, 1, 1)]))
    1.0
    """
    return get_min_belief(model, evident_event, evident_event)

def get_min_belief(model, evident_event, states):
    """
    >>> from asym_model import *
    >>> m = AsymmetricModel(0.1, 0.95)
    >>> get_min_belief(m, set([(1, 1, 1),(1, 1, 0)]), [(1, 1, 0)])
    0.09500000000000006
    >>> get_min_belief(m, set([(1, 1, 1),(1, 1, 0)]), [(1, 1, 1)])
    1.0
    >>> from simple_model import *
    >>> m = SimpleModel(0.1, 0.95)
    >>> get_min_belief(m, set([(1, 1, 1),(1, 1, 0)]), [(1, 1, 1)])
    0.95
    """

    dog_day_event = get_satisfying_states(model)
    
    p = 1.0

    for i in range(model.n_players):
        
        for x in states:
            
            this_p = min(conditional(model, i, evident_event, x), conditional(model, i, dog_day_event, x))
            
            if this_p < p:
                p = this_p
    
    return p

if __name__ == "__main__":
    import doctest
    doctest.testmod()
