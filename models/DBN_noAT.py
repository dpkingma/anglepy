import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.BNModel as BNModel
import anglepy.logpdfs as logpdfs
import math, inspect
from theano.tensor.shared_randomstreams import RandomStreams

def create(n_z, n_x, n_steps, n_batch, prior_sd=0.1):
	constr = (__name__, inspect.stack()[0][3], locals())
	
	A = np.ones((1, n_batch))
	
	def functions(w, z, x):
		
		def f_xi(zi, xi):
			pi = T.nnet.sigmoid(T.dot(w['wx'], zi) + T.dot(w['bx'], A)) # p = p(X=1)
			logpxi = - T.nnet.binary_crossentropy(pi, xi).sum(axis=0, keepdims=True) # logpxi = log p(X_i=x_i)
			return logpxi
		
		# Factors of X and Z
		logpx = 0
		logpz = 0
		sd = T.dot(T.exp(w['logsd']), A)
		for i in range(n_steps):
			if i == 0:
				logpz += logpdfs.standard_normal(z['z'+str(i)]).sum(axis=0, keepdims=True)
			if i > 0:
				mean = T.tanh(T.dot(w['wz'], z['z'+str(i-1)]) + T.dot(w['bz'], A))
				logpz += logpdfs.normal(z['z'+str(i)], mean, sd).sum(axis=0, keepdims=True)
			logpxi = f_xi(z['z'+str(i)], x['x'+str(i)])
			logpx += logpxi
		
		# joint() = logp(x,z,w) = logp(x|z) + logp(z) + logp(w) + C
		# This is a proper scalar function
		logpw = 0
		for i in w:
			logpw += logpdfs.normal(w[i], 0, prior_sd).sum() # logp(w)
		
		return logpw, logpx, logpz
	
	# Confabulate latent variables 'x' and 'z'
	def gen_xz(w, x, z):
		sd = np.dot(np.exp(w['logsd']), A)
		for i in range(n_steps):
			if not z.has_key('z'+str(i)):
				if i == 0:
					z['z'+str(i)] = np.random.standard_normal(size=(n_z, n_batch))
				else:
					mean = np.tanh(np.dot(w['wz'], z['z'+str(i-1)]) + np.dot(w['bz'], A))
					z['z'+str(i)] = mean + sd * np.random.standard_normal(size=(n_z, n_batch))
			
			pi = 1/(1+np.exp(-np.dot(w['wx'], z['z'+str(i)]) - np.dot(w['bx'], A)))
			if not x.has_key('x'+str(i)):
				x['x'+str(i)] = np.random.binomial(n=1,p=pi,size=pi.shape)
			
		return x, z, {}

	# Initial parameters
	def init_w(std = 1e-2):
		w = {}
		w['wz'] = np.random.normal(0, std, size=(n_z, n_z))
		w['bz'] = np.random.normal(0, std, size=(n_z, 1))
		w['logsd'] = np.random.normal(0, std, size=(n_z, 1))
		w['wx'] = np.random.normal(0, std, size=(n_x, n_z))
		w['bx'] = np.random.normal(0, std, size=(n_x, 1))
		return w
	
	def variables():
		# Define parameters 'w'
		w = {}
		w['wz'] = T.dmatrix('wz')
		w['bz'] = T.dmatrix('bz')
		w['logsd'] = T.dmatrix('logsd')
		w['wx'] = T.dmatrix('wx')
		w['bx'] = T.dmatrix('bx')
		
		# Define variables 'x' and 'z'
		z = {}
		x = {}
		for i in range(n_steps):
			z['z'+str(i)] = T.dmatrix('z'+str(i))
			x['x'+str(i)] = T.dmatrix('x'+str(i))
			
		return w, x, z
	
	theano_warning = 'raise'
	if n_steps == 1: theano_warning = 'warn'
	
	return BNModel.BNModel(constr, variables, functions, init_w, gen_xz, n_batch, theano_warning)
