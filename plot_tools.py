
import numpy as np
import pandas as pd

from config import *

def expand(df):
    """
    >>> df = pd.DataFrame([['Test',0,0.1],['Test',1,0.2],['Test',2,0.3],['Test',3,0.4]])
    >>> df = expand(df)
    >>> len(df) == sum(playN)
    True
    >>> abs(np.mean(df[df['Condition'] == 1]['Coordinate']) - 0.2) < 0.1
    True
    >>> df = pd.DataFrame([['Test',0,0.1],['Test',1,0.2],['Test',2,0.3],['Test',3,0.4],['Test2',0,0.5],['Test2',1,0.6],['Test2',2,0.7],['Test2',3,0.8]])
    >>> df = expand(df)
    >>> len(df) == sum(playN)*2
    True
    >>> abs(np.mean(df[(df['Model'] == 'Test2')&(df['Condition'] == 3)]['Coordinate']) - 0.8) < 0.1
    True
    """
    df = np.array(df)
    new = []
    for r in df:
        model = r[0]
        condition = r[1]
        N = playN[condition] 
        n = int(N * r[2])
        new += [[model, condition, 1]]*n + [[model, condition, 0]]*(N - n)
    new = pd.DataFrame(new, columns = ['Model', 'Condition', 'Coordinate'])
    return new

if __name__ == "__main__":
    import doctest
    doctest.testmod()
