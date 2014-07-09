import numpy as np
import matplotlib.pyplot as plt
import anglepy.ndict as ndict
import scipy.stats
import scipy.linalg
import time, sys
import random

log = {'acceptRate': [1],}

# Hybrid Monte Carlo sampler
# HMC move where x is a batch (rows=variables, columns=samples)
# NOTE: _stepsize can be a scalar OR a column vector with individual stepsizes
def hmc_step(fgrad, x0, _stepsize=1e-2, n_steps=20):
	
	# === INITIALIZE
	n_batch = x0.itervalues().next().shape[1]
	stepsize = (np.random.uniform(size=(1, n_batch)) < 0.5).astype(float)*2-1
	stepsize *= _stepsize
	
	if np.random.uniform() < 0.5:
		stepsize *= -1
	
	# Sample velocity
	vnew = {}
	for i in x0:
		vnew[i] = np.random.normal(size=x0[i].shape)
	
	# copy initial state 
	xnew = ndict.clone(x0)
	v0 = ndict.clone(vnew)
	
	# === LEAPFROG STEPS
	
	# Compute velocity at time (t + stepsize/2)
	# Compute position at time (t + stepsize)
	logpxz0, g = fgrad(xnew)
	for i in xnew:
		vnew[i] += 0.5 * stepsize * g[i]
		xnew[i] += stepsize * vnew[i]
	
	# Perform leapfrog steps
	for step in xrange(n_steps):
		#print 'hmc_step:', step
		logpxz, g = fgrad(xnew)
		for i in xnew:
			vnew[i] += stepsize * g[i]
			xnew[i] += stepsize * vnew[i]
	
	# Perform final half-step for velocity
	logpxz1, g = fgrad(xnew)
	for i in xnew:
		vnew[i] += 0.5 * stepsize * g[i]
	
	# === METROPOLIS-HASTINGS ACCEPT/REJECT
	
	# Compute old and new Hamiltonians

	k0 = 0
	k1 = 0
	for i in vnew:
		k0 += (v0[i]**2).sum(axis=0, keepdims=True)
		k1 += (vnew[i]**2).sum(axis=0, keepdims=True)
		
	h0 = -logpxz0 + 0.5 * k0
	h1 = -logpxz1 + 0.5 * k1
	
	#print logpxz0, k0, logpxz1, k1
	#print h0-h1
	
	# Perform Metropolis-Hasting step
	accept = np.exp(h0 - h1) >= np.random.uniform(size=h1.shape)
	accept = accept.astype(float)
	
	for i in xnew:
		accept2 = np.dot(np.ones((xnew[i].shape[0], 1)), accept)
		x0[i] = (1-accept2)*x0[i] + accept2 * xnew[i]
	
	# result: updated 'x0'
	logpxz = (1-accept)*logpxz0 + accept * logpxz1
	
	return logpxz, accept

# Auto-tuning HMC step with global stepsize
def hmc_step_autotune(n_steps=20, init_stepsize=1e-2, target=0.9):
	alpha = 1.02 #1.02
	max_factor = 1.25
	stepsize = [init_stepsize]
	steps = [0]
	def dostep(fgrad, x):
		logpxz, accept = hmc_step(fgrad, x, stepsize[0], n_steps)
		factor = min(max_factor, alpha**float(min(100, accept.size)))
		if accept.mean() < target:
			stepsize[0] /= factor
		else:
			stepsize[0] *= factor
		steps[0] += 1
		#print steps[0], 'stepsize:', accept.mean(), stepsize[0], accept.size
		return logpxz, accept.mean(), stepsize[0]
	return dostep

# Auto-tuning with individual stepsizes
def hmc_step_autotune_indiv(n_steps=20, init_stepsize=1e-2, target=0.9):
	alpha = 1.02 #1.02
	stepsize = [None]
	init = [False]
	steps = [0]
	def dostep(fgrad, x):
		if init[0] == False:
			n_batch = x.itervalues().next().shape[1]
			stepsize[0] = init_stepsize*np.ones((1, n_batch))
			init[0] = True
		logpxz, accept = hmc_step(fgrad, x, stepsize[0], n_steps)		
		stepsize[0] *= np.exp((alpha * (accept*2-1)))
		steps[0] += 1
		#print steps[0], 'stepsize:', accept.mean(), stepsize[0], accept.size
		return logpxz, accept.mean(), stepsize[0]
	return dostep

# Merge lists of z's and corresponding logpxz's into single matrices
def combine_samples(z_list, logpxz_list):
	n_list = len(logpxz_list)
	
	n_batch = logpxz_list[0].shape[1]
	
	# Stack logpxz values into one matrix
	for i in range(n_list):
		logpxz_list[i] = logpxz_list[i].reshape((n_batch, 1))
	logpxz = np.hstack(logpxz_list).reshape((1, n_batch*n_list))
	
	# Stack x samples into one ndict
	z = {}
	for i in z_list[0].keys():
		list = []
		for _z in z_list:
			list.append(_z[i].reshape((-1, 1)))
		z[i] = np.hstack(list).reshape((-1, n_batch*n_list))
	
	return z, logpxz

# Convenience function: run HMC automatically
def sample_standard_auto(z, fgrad, n_burnin, n_samples):
	hmc_dostep = hmc_step_autotune(n_steps=10, init_stepsize=1e-2, target=0.9)
	def dostep(_z):
		logpxz, _, _ = hmc_dostep(fgrad, _z)
		return logpxz
	
	# Burn-in phase
	for i in range(n_burnin):
		dostep(z)
	
	# Sample
	z_list = []
	logpxz_list = []
	for _ in range(n_samples):
		logpxz = dostep(z)
		#print 'logpxz:', logpxz
		logpxz_list.append(logpxz.copy())
		z_list.append(ndict.clone(z))
	
	z, logpxz = combine_samples(z_list, logpxz_list)
	
	return z, logpxz

