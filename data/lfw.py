import numpy as np
import os

path = os.path.dirname(__file__)+'/'

# shape: (13233, 31, 23, 3)
def load_lfw_small():
	data = np.load(path+'lfw_31x23.npy').swapaxes(1,3).swapaxes(2,3).reshape((-1, 31*23*3))
	data = data.astype('float16')/256.
	return data, (31, 23)

def load_lfw_big():
	data = np.load(path+'lfw_62x47.npy').swapaxes(1,3).swapaxes(2,3).reshape((-1, 62*47*3))
	data = data.astype('float16')/256.
	return data, (62, 47)

