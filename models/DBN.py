import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.logpdfs
import anglepy.BNModel as BNModel
import anglepy as ap

import math, inspect
from theano.tensor.shared_randomstreams import RandomStreams

class DBN(BNModel):
	
	def __init__(self, n_z, n_x, n_steps, prior_sd=0.1, data='binary'):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_z, self.n_x, self.n_steps = n_z, n_x, n_steps
		self.prior_sd = prior_sd
		self.data = data
		theano_warning = 'raise'
		if n_steps == 1: theano_warning = 'warn'
		
		super(DBN, self).__init__(theano_warning)
	
	def factors(self, w, x, z, A):
		
		if self.data == 'binary':
			def f_xi(zi, xi):
				pi = T.nnet.sigmoid(T.dot(w['wx'], zi) + T.dot(w['bx'], A)) # pi = p(X_i=1)
				logpxi = - T.nnet.binary_crossentropy(pi, xi).sum(axis=0, keepdims=True)# logpxi = log p(X_i=x_i)
				#logpxi = T.log(pi*xi+(1-pi)*(1-xi)).sum(axis=0, keepdims=True)
				return logpxi
		elif self.data == 'gaussian':
			def f_xi(zi, xi):
				x_mean = T.dot(w['wx'], zi) + T.dot(w['bx'], A)
				x_logvar = T.dot(2*w['logsdx'], A)
				return ap.logpdfs.normal2(xi, x_mean, x_logvar).sum(axis=0, keepdims=True)
		else: raise Exception()
			
		# Factors of X
		hidden = []
		logpx = 0
		sd = T.exp(w['logsd'])
		
		for i in range(self.n_steps):
			if i == 0:
				#_z = T.tanh(z['eps'+str(i)])
				_z = z['eps'+str(i)]
			else:
				_z = z['eps'+str(i)] * T.dot(sd, A)
				_z += T.tanh(T.dot(w['wz'], hidden[i-1]) + T.dot(w['bz'], A))
			hidden.append(_z)
			logpx += f_xi(_z, x['x'+str(i)])
		
		# Factors of Z
		logpz = 0
		for i in z:
			logpz += anglepy.logpdfs.standard_normal(z[i]).sum(axis=0, keepdims=True) # logp(z)
		
		# joint() = logp(x,z,w) = logp(x|z) + logp(z) + logp(w) + C
		# This is a proper scalar function
		logpw = 0
		for i in w:
			logpw += anglepy.logpdfs.normal(w[i], 0, self.prior_sd).sum() # logp(w)
		
		return logpw, logpx, logpz, {}

	# Confabulate hidden states 'z'
	def gen_xz(self, w, x, z, n_batch):
		
		A = np.ones((1, n_batch))
		
		# Factors of X
		_z = {}
		sd = np.dot(np.exp(w['logsd']), A)
		for i in range(self.n_steps):
			if not z.has_key('eps'+str(i)):
				z['eps'+str(i)] = np.random.standard_normal(size=(self.n_z, n_batch))
			
			if i == 0:
				_z['z'+str(i)] = z['eps'+str(i)]
			else:
				_z['z'+str(i)] = z['eps'+str(i)] * sd
				_z['z'+str(i)] += np.tanh(np.dot(w['wz'], _z['z'+str(i-1)]) + np.dot(w['bz'], A))
			if not x.has_key('x'+str(i)):
				if self.data == 'binary':
					pi = 1/(1+np.exp(-np.dot(w['wx'], _z['z'+str(i)]) - np.dot(w['bx'], A))) # pi = p(X_i=1)
					x['x'+str(i)] = np.random.binomial(n=1,p=pi,size=pi.shape)
				elif self.data == 'gaussian':
					x_mean = np.dot(w['wx'], _z['z'+str(i)]) + np.dot(w['bx'], A)
					x_logvar = np.dot(2*w['logsdx'], A)
					x['x'+str(i)] = np.random.normal(x_mean, np.exp(x_logvar/2))
				else: raise Exception()
		return x, z, _z
	
	# Extra function that computes z['eps'] given _z['z']
	def z_to_eps(self, w, _z, n_batch):
		A = np.ones((1, n_batch))
		sd = np.dot(np.exp(w['logsd']), A)
		z = {}
		for i in range(self.n_steps):
			if i == 0:
				z['eps'+str(i)] = _z['z'+str(i)]
			else:
				z['eps'+str(i)] = (_z['z'+str(i)] - np.tanh(np.dot(w['wz'], _z['z'+str(i-1)]) + np.dot(w['bz'], A))) / sd
		return z
	
	# Initial parameters
	def init_w(self, std = 1e-2):
		n_z, n_x = self.n_z, self.n_x
		w = {}
		w['wz'] = np.random.standard_normal(size=(n_z, n_z)) * std
		w['bz'] = np.random.standard_normal(size=(n_z, 1)) * std
		w['logsd'] = np.random.standard_normal(size=(n_z, 1)) * std
		w['wx'] = np.random.standard_normal(size=(n_x, n_z)) * std
		w['bx'] = np.random.standard_normal(size=(n_x, 1)) * std
		if self.data == 'gaussian':
			w['logsdx'] = np.random.standard_normal(size=(n_z, 1)) * std
		return w
	
	def variables(self):
		# Define parameters 'w'
		w = {}
		vars = ['wz','bz','logsd','wx','bx']
		if self.data == 'gaussian':
			vars.append('logsdx')
		for i in vars:
			w[i] = T.dmatrix(i)
		
		# Define variables 'x' and 'z'
		z = {}
		x = {}
		for i in range(self.n_steps):
			z['eps'+str(i)] = T.dmatrix('eps'+str(i))
			x['x'+str(i)] = T.dmatrix('x'+str(i))
			
		return w, x, z
	
	