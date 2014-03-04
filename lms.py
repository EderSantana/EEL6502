"""
LMS algorihtm
"""
import numpy as N
import gc

class LMS(object):
    def __init__(self, dataset, desired, delays,
            learning_rate=.01,
            learning_rate_mu = 0,
            mu=1, w=None, b=None):
    
        if w is None:
            self.w = N.random.randn(delays,1)
        else:
            self.w = w
        
        if b is None:
            self.b = 0.
        else:
            self.b = b

        self.dataset = dataset
        self.desired = desired
        self.delays = delays
        self.mu = mu
        self.learning_rate = learning_rate
        self.learning_rate_mu = learning_rate_mu

        self.prev_x = 0.
        self.error = N.asarray([])
        self.y = N.asarray([])
        self.w_track = N.zeros((delays, 1))
        self.alpha = N.zeros((self.delays,1))
        self.Xf =  N.zeros((self.delays,1))


    def fprop(self, X):
        """
        Forward propagte the input
        """

        y = N.dot(X.transpose(), self.w) + self.b

        return y

    def get_next_X(self, sample):
        """
        Get an input delay line
        """
        
        X  = N.zeros((self.delays,1))
        # TODO: Make the Gamma Filter work and delete this
        if self.learning_rate_mu==0.: #Just simple working solution for the regular delays
            for i in range(self.delays):
                i_sample = sample - i
                if i_sample>=0:
                    self.Xf[i] = self.dataset[i_sample]
                else:
                    self.Xf[i] = 0
        else:
            assert self.delays == 3
            if sample-2 >= 0:
                self.Xf[2] = (1-self.mu)*self.Xf[2] + self.mu*(1-self.mu)*self.Xf[1] + self.mu**2 * self.dataset[sample-2]
                self.Xf[1] = (1-self.mu)*self.Xf[1] + self.mu*self.dataset[sample-1]
            else:
                self.Xf[2] = (1-self.mu)*self.Xf[2]
                self.Xf[1] = (1-self.mu)*self.Xf[1]
            self.Xf[0] = self.dataset[sample]

        d = self.desired[sample]

        return d

    def sgd(self, sample):
        
        assert self.Xf.shape[0]==self.delays

        d = self.get_next_X(sample)
        y = self.fprop(self.Xf)
        e = d-y
        self.y = N.append(self.y, y)
        self.error = N.append(self.error, e)

        self.w_track = N.hstack( [self.w_track, self.w] )
        self.w = self.w + self.learning_rate*e*self.Xf/N.linalg.norm(self.Xf)

        self.b = self.b + self.learning_rate*e
        
        for i in range(1,self.delays):
            self.alpha[i] = (1-self.mu)*self.alpha[i-1] + \
            self.mu*self.alpha[i-1] + self.Xf[i-1] - self.Xf[i]

        self.mu = self.mu + self.learning_rate_mu * e * \
                N.dot(self.alpha.transpose(), self.w)
        
        if self.mu > 2 or self.mu<0:
            self.mu = N.mod(self.mu, 2)

    def train_lms(self):
        for i in range(len(self.dataset)):
            if N.mod(i,1000)==0:
                #print i
                gc.collect()
            self.sgd(i)

