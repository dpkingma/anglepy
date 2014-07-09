import numpy as np
import collections as C
import anglepy as ap
import anglepy.ndict as ndict
from anglepy.misc import lazytheanofunc

import theano
import theano.tensor as T


# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=10):
    new_y = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        new_y[y[i], i] = 1
    return new_y
def unbinarize_labels(y):
    return np.argmax(y,axis=0)

'''
Pseudo-Label
'''
class PL():
    
    def __init__(self, n_x, n_h, n_y, lr=0, nonlinear='softplus', valid_x=None, valid_y=None):
        print 'PL', n_x, n_h, n_y, lr, nonlinear
        if lr == 0: lr = 10. / n_h
        self.lr = lr
        self.fitted = False
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.nonlinear = nonlinear
        self.valid_x = valid_x
        self.valid_y = valid_y
        
        if self.nonlinear == 'softplus':
            def g(_x): return T.log(T.exp(_x) + 1)
        else:
            raise Exception()
        
        # Define Theano computational graph
        x, y, w1, b1, w2, b2, A = T.dmatrices('x', 'y', 'w1', 'b1', 'w2', 'b2', 'A')
        h1 = g(T.dot(w1, x) + T.dot(b1, A))
        h2 = g(T.dot(w2, h1) + T.dot(b2, A))
        p = T.nnet.softmax(h2.T).T
        logpy = (- T.nnet.categorical_crossentropy(p.T, y.T).T).reshape((1,-1))
        dlogpy_dw = T.grad(logpy.sum(), [w1, b1, w2, b2])
        H = T.nnet.categorical_crossentropy(p.T, p.T).T #entropy
        dH_dw = T.grad(H.sum(), [w1, b1, w2, b2])
        
        # Define functions to call
        self.f_p = theano.function([x, w1, b1, w2, b2, A], p)
        self.f_dlogpy_dw = theano.function([x, y, w1, b1, w2, b2, A], [logpy] + dlogpy_dw)
        self.f_dH_dw = theano.function([x, w1, b1, w2, b2, A], [H] + dH_dw)
        
    
    def fit(self, train_x, train_y, unlabeled_x):

        # Optimization params
        n_batch = 100
        n_its = 100000
        warmup = 100
        
        train_x = train_x.T
        unlabeled_x = unlabeled_x.T
        train_y = binarize_labels(train_y)
        n_tot_l = train_x.shape[1]
        n_tot_u = unlabeled_x.shape[1]
        
        # Initialize weights
        std = 0.01
        def rand(size):
            return np.random.normal(0, std, size=size)
        self.w1 = rand((self.n_h, self.n_x))
        self.b1 = rand((self.n_h, 1))
        self.w2 = rand((self.n_y, self.n_h))
        self.b2 = rand((self.n_y, 1))
        
        # Start optimization
        A = rand((1, n_batch))
        gw1_ss = np.zeros(self.w1.shape)
        gb1_ss = np.zeros(self.b1.shape)
        gw2_ss = np.zeros(self.w2.shape)
        gb2_ss = np.zeros(self.b2.shape)
        
        objective = []
        
        for it in range(n_its):
            # Get labeled set gradient
            idx_minibatch = np.random.randint(0, n_tot_l, n_batch)
            x_minibatch = train_x[:,idx_minibatch]
            y_minibatch = train_y[:,idx_minibatch]
            logpy, gw1, gb1, gw2, gb2 = self.f_dlogpy_dw(x_minibatch, y_minibatch, self.w1, self.b1, self.w2, self.b2, A)
            
            # Get unlabeled set gradient
            idx_minibatch = np.random.randint(0, n_tot_u, n_batch)
            x_minibatch = unlabeled_x[:,idx_minibatch]
            H, _gw1, _gb1, _gw2, _gb2 = self.f_dH_dw(x_minibatch, self.w1, self.b1, self.w2, self.b2, A)
            
            # Add losses (maximize logpy, minimize entropy, therefore substraction)
            beta = 10 # 
            t1 = 100. * beta
            t2 = 600. * beta
            af = -3
            if it < t1: alpha = 0
            elif it > t1 and it < t2: alpha = (it-t1)/(t2-t1) * af
            else: alpha = af
            
            gw1 += alpha*_gw1 
            gb1 += alpha*_gb1
            gw2 += alpha*_gw2
            gb2 += alpha*_gb2
            
            gw1_ss += gw1**2
            gb1_ss += gb1**2
            gw2_ss += gw2**2
            gb2_ss += gb2**2
            
            adagrad_reg = 1e-8
            if it > warmup:
                self.w1 += self.lr / np.sqrt(gw1_ss + adagrad_reg) * gw1
                self.b1 += self.lr / np.sqrt(gb1_ss + adagrad_reg) * gb1
                self.w2 += self.lr / np.sqrt(gw2_ss + adagrad_reg) * gw2
                self.b2 += self.lr / np.sqrt(gb2_ss + adagrad_reg) * gb2
            
            objective.append(np.average(logpy - H))
            
            # Check test set error
            if it%10000 == 0:
                if self.valid_x != None:
                    valid_err = 100-100*np.average(self.predict(self.valid_x) == self.valid_y)
                else:
                    valid_err = None
                    test_err = None
                    
                print it, np.average(np.array(objective)), valid_err
                objective = []
            
        
    def predict(self, x):
        p = self.predict_prob(x)
        return np.argmax(p, 1)
    
    def predict_prob(self, x):
        x = x.T
        n_batch = x.shape[1]
        A = np.ones((1,n_batch))
        p = self.f_p(x, self.w1, self.b1, self.w2, self.b2, A)
        return p.T
        
        