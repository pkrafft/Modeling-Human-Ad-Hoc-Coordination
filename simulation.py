
from player import *
from pbelief import *

from config import *

def get_offsets(length = 50):
    """
    >>> get_offsets(5)
    array([ 0.25,  0.5 ,  0.75])
    """
    return np.linspace(0,1,length)[1:-1]

def reward(payoffs, action, p):
    """
    >>> reward([2, -1, 1, 0], 1, 1)
    2
    >>> reward([2, -1, 1, 0], 1, 0.5)
    0.5
    >>> reward([2, -1, 1, 0], 0, 0.5)
    1.0
    """
    return action * p * payoffs[0] + action * (1-p) * payoffs[1] + (1 - action) * payoffs[2]

def compare(payoffs, verbose = False):
    """
    >>> compare([2,0,1,0.5], True)
    [0, 0, 0]
    [1, 0, 0]
    [1, 1, 0]
    [1, 1, 1]
    [[0.5, 1.1475845410628018, 'Private Heuristic'], [0.5, 1.0086956521739125, 'Pair Heuristic'], [0.5, 0.73333333333333339, 'Cognitive Strategy']]
    """
    
    totals = np.zeros(len(settings))

    p_star = optimal_threshold(payoffs)
    
    actions = [None for i in range(len(simulation_names))] # simulation models from config.py
    
    for i in range(len(settings)):

        for j in range(len(simulation_models)):

            actions[j] = simulation_models[j](worlds[i], 1 - players[i], settings[i], payoffs)
            totals[j] += reward(payoffs, actions[j], data[i]) # data from config.py

        if verbose:
            print actions
    
    totals -= payoffs[2]*len(settings)
    
    results = []
    for i in range(len(simulation_models)):
        results += [[p_star, totals[i], simulation_names[i]]]
    
    return results

if __name__ == "__main__":
    import doctest
    doctest.testmod()
