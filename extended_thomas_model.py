
import model
import utils
import numpy as np

class ExtendedThomasModel(model.Model):
    
    def __init__(self, dog_day_prob, message_prob, n_messengers = 1):
        self.dog_day_prob = dog_day_prob
        self.message_prob = message_prob
        self.n_messengers = n_messengers
        self.message_length = 5
        super(ExtendedThomasModel, self).__init__([(0,1)]*(1 + self.message_length*self.n_messengers), 2)
    
    def run(self, variables):
        """
        >>> m = ExtendedThomasModel(0.1, 0.25)
        >>> m.run((1,0,1,0,0,0))
        [(None, None, None, None, None), (1, None, 1, None, None)]
        >>> m = ExtendedThomasModel(0.1, 0.25, 2)
        >>> m.run((1,0,0,0,0,0,1,1,1,1,1))
        [(None, None, None, None, None, 1, 1, 1, 1, 1), (None, None, None, None, None, 1, 1, 1, 1, None)]
        """
        
        out = [[] for i in range(self.n_players)]
        
        dog_day = variables[0]
        for i in range(self.n_messengers):
            this_vars = variables[(1 + i*self.message_length):(1 + (i+1)*self.message_length)]
            this_out = self.messenger(dog_day, *this_vars)
            for j in range(self.n_players):
                out[j] += this_out[j]
        
        out = [tuple(out[i]) for i in range(self.n_players)]
        
        return out

    # encodes the generative model of the messenger
    # d indicates if it is a "dog day"
    def messenger(self, d, m0, m1, tp, wtm, wtp):

        out = []
        
        if m0 == 1: # visiting player 0
            if tp: # tell plan
                if m1 == 1: # visiting player 1 
                    out += [[d, m0, m1, wtm, wtp]]
                else:
                    out += [[d, m0, m1, None, None]]
            else:
                out += [[d, m0, None, None, None]]
        else:
            out += [[None, None, None, None, None]]
        
        if m1 == 1: # visiting player 1
            if wtm: # will tell message given to player 0
                if m0 == 1: # visiting player 0
                    if wtp: # will tell player 0 plan to player 1
                        out += [[d, m0, m1, tp, None]]
                    else:
                        out += [[d, m0, m1, None, None]]
                else:
                    out += [[d, m0, m1, None, None]]
            else:
                out += [[d, None, m1, None, None]]
                    
        else:
            out += [[None, None, None, None, None]]
        
        return out
    
    def evaluate(self, variables, log = False):
        """
        >>> m = ExtendedThomasModel(0.1, 0.25)
        >>> np.abs(m.evaluate((1,0,1,0,0,0)) - 0.1 * 0.75 * 0.25 * 0.5 * 0.5 * 0.5) < 1e-8
        True
        >>> np.abs(np.exp(m.evaluate((1,0,1,0,0,0), True)) - 0.1 * 0.75 * 0.25 * 0.5 * 0.5 * 0.5) < 1e-8
        True
        >>> m = ExtendedThomasModel(0.1, 0.25, 2)
        >>> np.abs(np.exp(m.evaluate((1,0,1,0,0,0,1,1,0,0,0), True)) - 0.1 * 0.75 * 0.25 * 0.5 * 0.5 * 0.25 * 0.25 * 0.5 * 0.5 * 0.5 * 0.5) < 1e-8
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

    def messenger_log_prob(self, m0, m1, tp, wtm, wtp):
        
        log_p = 0
        log_p += utils.log_bernoulli(m0, self.message_prob)
        log_p += utils.log_bernoulli(m1, self.message_prob)
        log_p += utils.log_bernoulli(tp, 0.5)
        log_p += utils.log_bernoulli(wtm, 0.5)
        log_p += utils.log_bernoulli(wtp, 0.5)
        
        return log_p
        
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
