
import model
import utils
import numpy as np

class SpeakerModel(model.Model):
    
    def __init__(self, delta, p):
        """
        >>> import utils
        >>> m = SpeakerModel(0.1, 0.25)
        >>> m.probs
        {(0, 1): 0.22500000000000001, (1, 0): 0.075000000000000025, (0, 0): 0.67500000000000004, (1, 1): 0.025000000000000012}
        >>> utils.print_partition(m.partitions)
        player 0
        0   [(0, 1)]
        1   [(1, 1)]
        None   [(0, 0), (1, 0)]
        player 1
        0   [(0, 1)]
        1   [(1, 1)]
        None   [(0, 0), (1, 0)]
        """
        self.delta = delta
        self.p = p
        self.message_length = 1
        super(SpeakerModel, self).__init__([(0,1)]*2, 2)
    
    def run(self, variables):
        """
        >>> m = SpeakerModel(0.1, 0.25)
        >>> m.run((1,0))
        [None, None]
        >>> m.run((0,1))
        [0, 0]
        """

        out = []
        
        d = variables[0]
        m0 = variables[1] 
        
        if m0 == 1: # announcement is made over speaker
            out += [(d), (d)]
        else:
            out = [(None), (None)]
            
        return out

    def evaluate(self, variables, log = False):
        """
        >>> m = SpeakerModel(0.1, 0.25)
        >>> np.abs(m.evaluate((1,0)) - 0.1 * 0.75) < 1e-8
        True
        >>> np.abs(np.exp(m.evaluate((1,1), True)) - 0.1 * 0.25) < 1e-8
        True
        """
        
        log_p = 0
        
        log_p += utils.log_bernoulli(variables[0], self.delta)
        log_p += utils.log_bernoulli(variables[1], self.p)

        if log:
            return log_p
        else:
            return np.exp(log_p)
            
if __name__ == "__main__":
    import doctest
    doctest.testmod()
