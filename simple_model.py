
import model
import utils
import numpy as np

class SimpleModel(model.Model):
    
    def __init__(self, delta, p):
        """
        >>> import utils
        >>> m = SimpleModel(0.1, 0.25)
        >>> m.probs
        {(0, 1, 1): 0.056250000000000001, (1, 1, 0): 0.018750000000000013, (1, 0, 0): 0.056250000000000029, (0, 0, 1): 0.16875000000000001, (1, 0, 1): 0.018750000000000013, (0, 0, 0): 0.50624999999999998, (0, 1, 0): 0.16874999999999998, (1, 1, 1): 0.0062500000000000021}
        >>> utils.print_partition(m.partitions)
        player 0
        0   [(0, 1, 0), (0, 1, 1)]
        1   [(1, 1, 0), (1, 1, 1)]
        None   [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
        player 1
        0   [(0, 0, 1), (0, 1, 1)]
        1   [(1, 0, 1), (1, 1, 1)]
        None   [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
        """
        self.delta = delta
        self.p = p
        self.message_length = 1
        super(SimpleModel, self).__init__([(0,1)]*3, 2)
    
    def run(self, variables):
        """
        >>> m = SimpleModel(0.1, 0.25)
        >>> m.run((1,0,1))
        [None, 1]
        >>> m.run((0,1,1))
        [0, 0]
        """

        out = []

        d = variables[0]
        m0 = variables[1]
        m1 = variables[2]
        
        if m0 == 1:
            out += [(d)]
        else:
            out += [(None)]
        
        if m1 == 1:
            out += [(d)]
        else:
            out += [(None)]
        
        return out

    def evaluate(self, variables, log = False):
        """
        >>> m = SimpleModel(0.1, 0.25)
        >>> np.abs(m.evaluate((1,0,1)) - 0.1 * 0.75 * 0.25) < 1e-8
        True
        >>> np.abs(np.exp(m.evaluate((1,0,0), True)) - 0.1 * 0.75 * 0.75) < 1e-8
        True
        """
        
        log_p = 0
        
        log_p += utils.log_bernoulli(variables[0], self.delta)
        log_p += utils.log_bernoulli(variables[1], self.p)
        log_p += utils.log_bernoulli(variables[2], self.p)

        if log:
            return log_p
        else:
            return np.exp(log_p)
            
if __name__ == "__main__":
    import doctest
    doctest.testmod()
