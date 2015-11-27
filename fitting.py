
from config import *

def predict(model, pars, name, err_vals, predictions):
    """
    >>> err_vals = []
    >>> predictions = [[],[],[]]
    >>> predict(lambda w,x,y,z: 0 if w == 1 else -1, [0,1,0], 'Test', err_vals, predictions)
    Test 1 [0 0 0 0] 1.5220153475
    >>> err_vals
    [['Test', 1, 1.5220153474993583]]
    >>> predictions
    [['Test', 'Test', 'Test', 'Test'], [0, 1, 2, 3], [0, 0, 0, 0]]
    """

    best_fit = None
    best_value = (None, np.inf)
    
    for par in pars:
    
        vals = np.array([None for i in range(len(settings))])
        for i in range(len(settings)):
            vals[i] = model(par, worlds[i], players[i], settings[i])

        error = sum((vals - data)**2)
        
        if error < best_value[1]:
            best_fit = par
            best_value = (vals, error)

    print name, best_fit, best_value[0], best_value[1]
    
    err_vals += [[name, best_fit, best_value[1]]]
    predictions[0] += [name]*len(settings)
    predictions[1] += range(len(settings))
    predictions[2] += list(best_value[0])

if __name__ == "__main__":
    import doctest
    doctest.testmod()
