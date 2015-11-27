
import itertools
import info_partition

class Model(object):

    def __init__(self, support, n_players):
        self.support = support
        self.n_players = n_players
        self.statespace = list(self.enumerate_statespace())
        self.partitions = info_partition.generate_partitions(self, n_players)
        self.graphs = info_partition.partitions_to_graphs(self.partitions)
        self.cache_state_probs()
    
    def enumerate_statespace(self):
        """
        >>> m = Model([(0,1),(0,1,2),(0,1)], 2)
        >>> list(m.enumerate_statespace())
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)]
        """
        return itertools.product(*self.support)

    # run the generative model of the world at a particular element of the statespace
    def run(self, x):
        return [None]*self.n_players
    
    # evaluate the probability (measure) of a particular element of the statespace
    def evaluate(self, x):
        return None

    def cache_state_probs(self):
        """
        >>> from simple_model import *
        >>> m = SimpleModel(0.1, 0.25)
        >>> m.probs
        {(0, 1, 1): 0.056250000000000001, (1, 1, 0): 0.018750000000000013, (1, 0, 0): 0.056250000000000029, (0, 0, 1): 0.16875000000000001, (1, 0, 1): 0.018750000000000013, (0, 0, 0): 0.50624999999999998, (0, 1, 0): 0.16874999999999998, (1, 1, 1): 0.0062500000000000021}
        """
        self.probs = dict(zip(self.statespace, [self.evaluate(x) for x in self.statespace]))
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
