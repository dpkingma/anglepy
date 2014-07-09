import numpy as np
import collections as C
import anglepy as ap
import anglepy.ndict as ndict
from anglepy.misc import lazytheanofunc

import svmlight

'''
To install pysvmlight (on MAC):
1) cd to pysvmlight dir
> export CFLAGS=-Qunused-arguments
> export CPPFLAGS=-Qunused-arguments
> chmod +x setup.py
> ./setup.py build
> sudo ./setup.py install
'''

'''
===> Example from pysvmlight doc:
# train a model based on the data
model = svmlight.learn(training_data, type='classification', verbosity=0)

# model data can be stored in the same format SVM-Light uses, for interoperability
# with the binaries.
svmlight.write_model(model, 'my_model.dat')

# classify the test data. this function returns a list of numbers, which represent
# the classifications.
predictions = svmlight.classify(model, test_data)
for p in predictions:
    print '%.8f' % p
'''

# Assumes:
# x is a matrix with one datapoint per row
def toSVMLightFeatures(x):
    _x = []
    for i in range(x.shape[0]):
        features = []
        for j in range(x.shape[1]):
            features.append((j+1, x[i,j]))
        _x.append(features)
    return _x

class TSVM():
    
    def __init__(self, C=100, rbf_gamma=0):
        self.C = C
        self.rbf_gamma = rbf_gamma
        self.fitted = False
    
    def fit(self, train_x, train_y, unlabeled_x=None):
        if self.rbf_gamma == 0:
            self.rbf_gamma = 1./train_x.shape[1]
        n_y = np.max(train_y)+1
        self.models = []
        feats = toSVMLightFeatures(train_x)
        if unlabeled_x != None:
            feats_unlabeled = toSVMLightFeatures(unlabeled_x)
        for i in range(n_y):
            train_y_binary = (train_y==i)*2-1
            input = []
            for i in range(len(feats)):
                input.append((train_y_binary[i], feats[i]))
            for i in range(len(feats_unlabeled)):
                input.append((0, feats_unlabeled[i]))
            _model = svmlight.learn(input, type='classification', kernel='rbf', C=self.C, rbf_gamma=self.rbf_gamma)
            self.models.append(_model)
        self.fitted = True
    
    def predict(self, x):
        if not self.fitted:
            raise Exception('Not fitted yet')
        if len(self.models) < 1:
            raise Exception("len(self.models) < 1")
        
        feats = toSVMLightFeatures(x)
        input = []
        for i in range(len(feats)):
            input.append((0, feats[i]))
        
        predictions = []
        for i in range(len(self.models)):
            predictions.append(np.array(svmlight.classify(self.models[i], input)))
        predictions = np.argmax(np.vstack(tuple(predictions)), axis=0)
        
        return predictions

    pass
