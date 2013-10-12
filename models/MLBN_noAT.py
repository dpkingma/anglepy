import numpy as np
import theano
import theano.tensor as T
import collections as C
import bnmodels.BNModel as BNModel
import bnmodels.logpdfs as logpdfs
import math, inspect

def create(n_hidden, n_output, n_batch, prior_sd=1):
	constr = (__name__, inspect.stack()[0][3], locals())
	
	A = np.ones((1, n_batch))
	
	def functions(w, z, x):
		
		# Define logp(z0)
		logpz = logpdfs.standard_normal(z['z0']).sum(axis=0)
		
		# Define logp(z_{n+1}|z_n)
		z_prev = z['z0']
		for i in range(1, len(n_hidden)):
			mean = T.tanh(T.dot(w['w'+str(i)], z_prev) + T.dot(w['b'+str(i)], A))
			sd = T.dot(T.abs_(w['logsd'+str(i)]), A)
			logpz += logpdfs.normal(z['z'+str(i)], mean, sd).sum(axis=0)
			z_prev = z['z'+str(i)]
		
		# logp(x|z_{last})	
		p = 1e-3 + (1-2e-3) * T.nnet.sigmoid(T.dot(w['wout'], z_prev) + T.dot(w['bout'], A)) # p = p(X=1)
		logpx = - T.nnet.binary_crossentropy(p, x['x'])
		logpx = T.dot(np.ones((1, n_output)), logpx)
		
		# joint() = logp(x,z,w) = logp(x,z|w) + logp(w) = logpxz + logp(w) + C
		# This is a proper scalar function
		logpw = 0
		for i in w:
			logpw += logpdfs.normal(w[i], 0, prior_sd).sum() # logp(w)
		
		return logpx, logpz, logpw
		
	def variables():
		# Define parameters 'w'
		w = {}
		for i in range(1, len(n_hidden)):
			w['w'+str(i)] = T.dmatrix('w'+str(i))
			w['b'+str(i)] = T.dmatrix('b'+str(i))
			w['logsd'+str(i)] = T.dmatrix('logsd'+str(i))
		w['wout'] = T.dmatrix('wout')
		w['bout'] = T.dmatrix('bout')
		
		# Define latent variables 'z'
		z = {}
		for i in range(len(n_hidden)):
			z['z'+str(i)] = T.dmatrix('z'+str(i))
		
		# Define observed variables 'x'
		x = {}
		x['x'] = T.dmatrix('x')
		
		return w, z, x
	
	# Confabulate latent variables
	def gen_xz(w, x, z):
		size = (n_hidden[0], n_batch)
		if not z.has_key('z0'):
			z['z0'] = np.random.standard_normal(size=size)
		for i in range(1, len(n_hidden)):
			if not z.has_key('z'+str(i)):
				sd = np.dot(np.absolute(w['logsd'+str(i)]), A)
				mean = np.tanh(np.dot(w['w'+str(i)], z['z'+str(i-1)]) + np.dot(w['b'+str(i)], A))
				size = (n_hidden[i], n_batch)
				z['z'+str(i)] = np.random.normal(mean, sd, size=size)
		
		if not x.has_key('x'):
			p = 1e-3 + (1-2e-3) * T.nnet.sigmoid(T.dot(w['wout'], z['z'+str(len(n_hidden)-1)]) + T.dot(w['bout'], A))
			x['x'] = np.random.binomial(n=1,p=p)
		
		return x, z, {}
	
	def init_w(std):
		w = {}
		for i in range(1, len(n_hidden)):
			w['w'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], n_hidden[i-1]))
			w['b'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], 1))
			w['logsd'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], 1))
		w['wout'] = np.random.normal(0, std, size=(n_output, n_hidden[-1]))
		w['bout'] = np.random.normal(0, std, size=(n_output, 1))
		return w
		
	return BNModel.BNModel(constr, variables, functions, init_w, gen_xz, n_batch)
