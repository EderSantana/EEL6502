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
        Xf = N.zeros((self.delays,1))

        for i in range(self.delays):
            i_sample = sample - i
            if i_sample>=0:
                X[i]  = self.dataset[i_sample]
                Xf[i] = (1-self.mu)*self.prev_x + self.mu*X[i]
                #self.dataset[i_sample]
                self.prev_x = Xf[i]
            else:
                X[i]  = 0.
                Xf[i] = 0.
        d = self.desired[sample]

        return Xf, X, d

    def sgd(self, sample):

        Xf, X, d = self.get_next_X(sample)
        y = self.fprop(Xf)
        e = d-y
        self.y = N.append(self.y, y)
        self.error = N.append(self.error, e)

        self.w_track = N.hstack( [self.w_track, self.w] )
        self.w = self.w + self.learning_rate*e*Xf#/N.linalg.norm(X)

        self.b = self.b + self.learning_rate*e
        
        for i in range(1,self.delays):
            self.alpha[i] = (1-self.mu)*self.alpha[i-1] + \
            self.mu*self.alpha[i-1] + Xf[i-1] - Xf[i]

        self.mu = self.mu + self.learning_rate_mu * e * \
                N.dot(self.alpha.transpose(), self.w)
        
        if self.mu > 2 or self.mu<0:
            self.mu = N.mod(self.mu, 2)

    def train_lms(self):
        for i in range(len(self.dataset)):
            if N.mod(i,1000)==0:
                print i
                gc.collect()
            self.sgd(i)

