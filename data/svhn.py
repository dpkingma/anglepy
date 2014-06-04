import numpy as np
import scipy.io
import os

# from http://cs.nyu.edu/~roweis/data.html
path = os.environ['ML_DATA_PATH']

def load_numpy(toFloat=True, binarize_y=False):
    train = scipy.io.loadmat(path+'/svhn/train_32x32.mat')
    train_x = train['X'].swapaxes(0,1).T.reshape((train['X'].shape[3], -1)).T
    train_y = train['y'].reshape((-1)) - 1
    test = scipy.io.loadmat(path+'/svhn/test_32x32.mat')
    test_x = test['X'].swapaxes(0,1).T.reshape((test['X'].shape[3], -1)).T
    test_y = test['y'].reshape((-1)) - 1
    if toFloat:
        train_x = train_x.astype('float16')/256.
        test_x = test_x.astype('float16')/256.
    if binarize_y:
        train_y = binarize_labels(train_y)
        test_y = binarize_labels(test_y)

    return train_x, train_y, test_x, test_y 

# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=10):
    new_y = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        new_y[y[i], i] = 1
    return new_y
