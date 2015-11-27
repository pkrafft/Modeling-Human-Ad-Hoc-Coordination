

def generate_partitions(model, n_players):
    """
    >>> from asym_model import *
    >>> import utils
    >>> m = AsymmetricModel(0.1, 0.25)
    >>> utils.print_partition(m.partitions)
    player 0
    (1, None)   [(1, 1, 0), (1, 1, 1)]
    (0, None)   [(0, 1, 0), (0, 1, 1)]
    (None, None)   [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
    player 1
    (0, 1)   [(0, 1, 1)]
    (1, 0)   [(1, 0, 1)]
    (0, 0)   [(0, 0, 1)]
    (1, 1)   [(1, 1, 1)]
    (None, None)   [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    """
    
    partitions = [{} for i in range(n_players)]

    for x in model.statespace:

        obs = model.run(x)
        
        for i in range(n_players):
            
            if obs[i] in partitions[i]:
                partitions[i][obs[i]] += [x]
            else:
                partitions[i][obs[i]] = [x]

    return partitions

def partitions_to_graphs(partitions):
    """
    >>> partitions = [{0:[(0,0),(0,1)],1:[(1,0),(1,1)]}, {0:[(0,0),(1,0)],1:[(0,1)],(1,1):[(1,1)]}]
    >>> partitions_to_graphs(partitions)
    [{(0, 1): [(0, 0), (0, 1)], (1, 0): [(1, 0), (1, 1)], (0, 0): [(0, 0), (0, 1)], (1, 1): [(1, 0), (1, 1)]}, {(0, 1): [(0, 1)], (1, 0): [(0, 0), (1, 0)], (0, 0): [(0, 0), (1, 0)], (1, 1): [(1, 1)]}]
    """
    
    graphs = [{} for i in range(len(partitions))]
    
    for i in range(len(graphs)):

        for o in partitions[i]:

            for x in partitions[i][o]:

                graphs[i][x] = [y for y in partitions[i][o]]

    return graphs
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()

