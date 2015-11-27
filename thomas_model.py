
import model
import utils
import numpy as np

class ThomasModel(model.Model):
    
    def __init__(self, dog_day_prob, message_prob, n_messengers = 1):
        """
        >>> m = ThomasModel(0.1, 0.25)
        >>> m.probs
        {(1, 1, 1, 0, 0): 0.0015625000000000003, (1, 0, 1, 1, 0): 0.0046874999999999981, (1, 1, 0, 1, 1): 0.0046874999999999981, (1, 0, 0, 0, 0): 0.014062500000000011, (0, 0, 0, 0, 1): 0.12656249999999999, (1, 0, 1, 0, 0): 0.0046874999999999981, (0, 0, 1, 1, 1): 0.042187499999999989, (1, 1, 1, 1, 0): 0.0015625000000000003, (0, 1, 0, 1, 1): 0.042187499999999989, (1, 0, 0, 1, 0): 0.014062500000000011, (0, 0, 1, 0, 1): 0.042187499999999989, (0, 1, 1, 1, 0): 0.014062500000000011, (0, 0, 1, 0, 0): 0.042187499999999989, (1, 1, 1, 0, 1): 0.0015625000000000003, (0, 1, 0, 0, 0): 0.042187499999999989, (1, 0, 1, 1, 1): 0.0046874999999999981, (1, 1, 0, 1, 0): 0.0046874999999999981, (1, 1, 0, 0, 0): 0.0046874999999999981, (1, 0, 1, 0, 1): 0.0046874999999999981, (0, 0, 1, 1, 0): 0.042187499999999989, (0, 1, 1, 0, 0): 0.014062500000000011, (0, 0, 0, 1, 0): 0.12656249999999999, (1, 0, 0, 0, 1): 0.014062500000000011, (0, 0, 0, 0, 0): 0.12656249999999999, (1, 1, 1, 1, 1): 0.0015625000000000003, (0, 0, 0, 1, 1): 0.12656249999999999, (1, 0, 0, 1, 1): 0.014062500000000011, (0, 1, 0, 0, 1): 0.042187499999999989, (0, 1, 1, 0, 1): 0.014062500000000011, (0, 1, 0, 1, 0): 0.042187499999999989, (1, 1, 0, 0, 1): 0.0046874999999999981, (0, 1, 1, 1, 1): 0.014062500000000011}
        >>> utils.print_partition(m.partitions)
        player 0
        (1, 1, None, None)   [(1, 1, 0, 0, 0), (1, 1, 0, 0, 1), (1, 1, 1, 0, 0), (1, 1, 1, 0, 1)]
        (1, 1, 0, None)   [(1, 1, 0, 1, 0), (1, 1, 0, 1, 1)]
        (0, 1, 0, None)   [(0, 1, 0, 1, 0), (0, 1, 0, 1, 1)]
        (None, None, None, None)   [(0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0, 1, 0), (0, 0, 0, 1, 1), (0, 0, 1, 0, 0), (0, 0, 1, 0, 1), (0, 0, 1, 1, 0), (0, 0, 1, 1, 1), (1, 0, 0, 0, 0), (1, 0, 0, 0, 1), (1, 0, 0, 1, 0), (1, 0, 0, 1, 1), (1, 0, 1, 0, 0), (1, 0, 1, 0, 1), (1, 0, 1, 1, 0), (1, 0, 1, 1, 1)]
        (0, 1, 1, 1)   [(0, 1, 1, 1, 1)]
        (1, 1, 1, 1)   [(1, 1, 1, 1, 1)]
        (0, 1, 1, 0)   [(0, 1, 1, 1, 0)]
        (0, 1, None, None)   [(0, 1, 0, 0, 0), (0, 1, 0, 0, 1), (0, 1, 1, 0, 0), (0, 1, 1, 0, 1)]
        (1, 1, 1, 0)   [(1, 1, 1, 1, 0)]
        player 1
        (1, 0, 1, None)   [(1, 0, 1, 0, 0), (1, 0, 1, 0, 1), (1, 0, 1, 1, 0), (1, 0, 1, 1, 1)]
        (0, 0, 1, None)   [(0, 0, 1, 0, 0), (0, 0, 1, 0, 1), (0, 0, 1, 1, 0), (0, 0, 1, 1, 1)]
        (None, None, None, None)   [(0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0, 1, 0), (0, 0, 0, 1, 1), (0, 1, 0, 0, 0), (0, 1, 0, 0, 1), (0, 1, 0, 1, 0), (0, 1, 0, 1, 1), (1, 0, 0, 0, 0), (1, 0, 0, 0, 1), (1, 0, 0, 1, 0), (1, 0, 0, 1, 1), (1, 1, 0, 0, 0), (1, 1, 0, 0, 1), (1, 1, 0, 1, 0), (1, 1, 0, 1, 1)]
        (0, 1, 1, 1)   [(0, 1, 1, 1, 1)]
        (1, 1, 1, 1)   [(1, 1, 1, 1, 1)]
        (1, 1, 1, None)   [(1, 1, 1, 0, 0), (1, 1, 1, 1, 0)]
        (1, 1, 1, 0)   [(1, 1, 1, 0, 1)]
        (0, 1, 1, 0)   [(0, 1, 1, 0, 1)]
        (0, 1, 1, None)   [(0, 1, 1, 0, 0), (0, 1, 1, 1, 0)]
        """
        self.dog_day_prob = dog_day_prob
        self.message_prob = message_prob
        self.n_messengers = n_messengers
        self.message_length = 4
        super(ThomasModel, self).__init__([(0,1)]*(1 + self.message_length*self.n_messengers), 2)

    def run(self, variables):
        """
        >>> m = ThomasModel(0.1, 0.25)
        >>> m.run((1,0,1,0,0))
        [(None, None, None, None), (1, 0, 1, None)]
        >>> m.run((1,0,1,1,0))
        [(None, None, None, None), (1, 0, 1, None)]
        >>> m.run((1,1,0,0,0))
        [(1, 1, None, None), (None, None, None, None)]
        >>> m.run((1,1,0,1,0))
        [(1, 1, 0, None), (None, None, None, None)]
        >>> m.run((1,1,1,1,0))
        [(1, 1, 1, 0), (1, 1, 1, None)]
        >>> m = ThomasModel(0.1, 0.25, 2)
        >>> m.run((1,1,1,1,0,0,1,0,0))
        [(1, 1, 1, 0, None, None, None, None), (1, 1, 1, None, 1, 0, 1, None)]
        >>> m.run((1,0,0,0,0,1,1,1,1))
        [(None, None, None, None, 1, 1, 1, 1), (None, None, None, None, 1, 1, 1, 1)]
        """

        out = [[] for i in range(self.n_players)]

        dog_day = variables[0]
        for i in range(self.n_messengers):
            this_out = self.messenger(dog_day, *variables[(1 + i*self.message_length):(1 + (i+1)*self.message_length)])
            for j in range(self.n_players):
                out[j] += this_out[j]
        
        out = [tuple(out[i]) for i in range(self.n_players)]
        
        return out

    # encodes the generative model of the messenger
    # d indicates if it is a "dog day"
    def messenger(self, d, m0, m1, tp, wtp):

        out = []
        
        if m0 == 1: # visiting player 0
            if tp == 1: # tell plan
                if m1 == 1: # visiting player 1
                    out += [[d, m0, m1, wtp]]
                else:
                    out += [[d, m0, m1, None]]
            else:
                out += [[d, m0, None, None]]
        else:
            out += [[None, None, None, None]]
        
        if m1 == 1: # visiting player 1
            if m0 == 1 and wtp == 1: # will tell player 0 plan to player 1, if it exists
                out += [[d, m0, m1, tp]]
            else:
                out += [[d, m0, m1, None]]
        else:
            out += [[None, None, None, None]]
        
        return out
    
    def evaluate(self, variables, log = False):
        """
        >>> m = ThomasModel(0.1, 0.25)
        >>> np.abs(m.evaluate((1,0,1,0,0)) - 0.1 * 0.75 * 0.25 * 0.5 * 0.5) < 1e-8
        True
        >>> np.abs(np.exp(m.evaluate((1,0,1,0,0), True)) - 0.1 * 0.75 * 0.25 * 0.5 * 0.5) < 1e-8
        True
        >>> m = ThomasModel(0.1, 0.25, 2)
        >>> np.abs(np.exp(m.evaluate((1,0,1,0,0,1,1,0,0), True)) - 0.1 * 0.75 * 0.25 * 0.5 * 0.5 * 0.25 * 0.25 * 0.5 * 0.5) < 1e-8
        True
        """
        
        log_p = 0
        
        log_p += utils.log_bernoulli(variables[0], self.dog_day_prob)
        for i in range(self.n_messengers):
            log_p += self.messenger_log_prob(*variables[(1 + i*self.message_length):(1 + (i+1)*self.message_length)])
        
        if log:
            return log_p
        else:
            return np.exp(log_p)

    def messenger_log_prob(self, m0, m1, tp, wtp):
        
        log_p = 0
        log_p += utils.log_bernoulli(m0, self.message_prob)
        log_p += utils.log_bernoulli(m1, self.message_prob)
        log_p += utils.log_bernoulli(tp, 0.5)
        log_p += utils.log_bernoulli(wtp, 0.5)
        
        return log_p
        
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
