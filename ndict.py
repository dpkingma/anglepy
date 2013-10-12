import collections
import numpy as np
import numpy.linalg
import collections as C

def getCols(d, ifrom, ito):
	result = {}
	for i in d:
		result[i] = d[i][:,ifrom:ito]
	return result

def sum(ds):
	d = ds[0]
	for i in range(1, len(ds)):
		for j in ds[i]:
			d[j] += ds[i][j]
	return d

# ds: multiple dictionaries
# result: cols ifrom to ito from 'ds' 
def getcols_multiple(ds, ifrom, ito):
	result = []
	for d in ds:
		result.append(getCols(d, ifrom, ito))
	return result

def setCols(d, ifrom, ito, x):
	for i in d:
		d[i][:,ifrom:ito] = x[i]

def mapCols(d, func, ifrom, ito):
	x = getCols(d, ifrom, ito)
	func(x)
	setCols(d, ifrom, ito, x)

def size(d):
	result = 0
	for i in d:
		result += d[i].size
	return result

def cloneZeros(d):
	result = {}
	for i in d:
		result[i] = np.zeros(d[i].shape)
	return result

def cloneOnes(d):
	result = {}
	for i in d:
		result[i] = np.zeros(d[i].shape)
	return result

def clone(d):
	result = {}
	for i in d:
		result[i] = d[i].copy()
	return result

# d: dict
# result: 1-dim ndarray
def flatten(d):
	d = ordered(d)
	result = np.zeros((size(d)))
	pointer = 0
	for i in d:
		result[pointer:pointer+d[i].size] = d[i].reshape((d[i].size))
		pointer += d[i].size
	return result

# ds: multiple dictionaries
# result: 'ds' flattened into 1-dim array
def flatten_multiple(ds):
	result = []
	for d in ds:
		result.append(flatten(d))
	return np.concatenate(result)
	
# flat: 1-dim ndarray
# d: dict with certain names/shapes
# result: dict with same shape as 'd' and values of 'flat'
def unflatten(flat, d):
	d = ordered(d)
	result = {}
	pointer = 0
	for i in d:
		result[i] = flat[pointer:pointer+d[i].size].reshape(d[i].shape)
		pointer += d[i].size
	return result

def unflatten_multiple(flat, ds):
	result = []
	pointer = 0
	for d in ds:
		result.append(unflatten(flat[pointer:pointer+size(d)], d))
		pointer += size(d)
	return result

# Merge ndicts to one ndict
# ds: ndicts to merge
# prefixes: prefixes of new keys
def merge(ds, prefixes):
	result = {}
	for i in range(len(ds)):
		for j in ds[i]:
			result[prefixes[i]+j] = ds[i][j]
	return result

# Inverse of merge(.): unmerge ndict
# d: merged ndict
# prefixes: prefixes	
def unmerge(d, prefixes):
	results = [{} for _ in range(len(prefixes))]
	for key in d:
		for j in range(len(prefixes)):
			prefix = prefixes[j]
			if key[:len(prefix)] == prefix:
				results[j][key[len(prefix):]] = d[key]
	return results

# Get shapes of elements of d as a dict	
def getShapes(d):
	shapes = {}
	for i in d:
		shapes[i] = d[i].shape
	return shapes

# Set shapes of elements of d
def setShapes(d, shapes):
	result = {}
	for i in d:
		result[i] = d[i].reshape(shapes[i])
	return result

def p(d):
	for i in d: print i+'\n', d[i]

def pNorm(d):
	for i in d: print i, numpy.linalg.norm(d[i])

def pShape(d):
	for i in d: print i, d[i].shape

def hasNaN(d):
	result = False
	for i in d: result = result or np.isnan(d[i]).any()
	return result

def savetext(d, name):
	for i in d: np.savetxt(file('debug_'+name+'.txt', 'w'), d[i])
	
def ordered(d):
	return C.OrderedDict(sorted(d.items()))
