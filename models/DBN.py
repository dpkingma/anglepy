import numpy as np
import theano
import theano.tensor as T
import collections as C
import bnmodels.logpdfs
import bnmodels.BNModel as BNModel
import math, inspect
from theano.tensor.shared_randomstreams import RandomStreams

class DBN(BNModel.BNModel):
	
	def __init__(self, n_z, n_x, n_steps, n_batch, prior_sd=0.1):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_z, self.n_x, self.n_steps, self.n_batch = n_z, n_x, n_steps, n_batch
		self.prior_sd = prior_sd
		
		theano_warning = 'raise'
		if n_steps == 1: theano_warning = 'warn'
		
		super(DBN, self).__init__(n_batch, theano_warning)
	
	
	def factors(self, w, z, x):
		A = np.ones((1, self.n_batch))
	
		def f_xi(zi, xi):
			pi = T.nnet.sigmoid(T.dot(w['wx'], zi) + T.dot(w['bx'], A)) # pi = p(X_i=1)
			logpxi = - T.nnet.binary_crossentropy(pi, xi).sum(axis=0, keepdims=True)# logpxi = log p(X_i=x_i)
			#logpxi = T.log(pi*xi+(1-pi)*(1-xi)).sum(axis=0, keepdims=True)
			return logpxi
		
		# Factors of X
		hidden = []
		logpx = 0
		sd = T.exp(w['logsd'])
		
		for i in range(self.n_steps):
			if i == 0:
				_z = T.tanh(z['eps'+str(i)])
			else:
				_z = z['eps'+str(i)] * T.dot(sd, A)
				_z += T.tanh(T.dot(w['wz'], hidden[i-1]) + T.dot(w['bz'], A))
			hidden.append(_z)
			logpx += f_xi(_z, x['x'+str(i)])
		
		# Factors of Z
		logpz = 0
		for i in z:
			logpz += bnmodels.logpdfs.standard_normal(z[i]).sum(axis=0, keepdims=True) # logp(z)
		
		# joint() = logp(x,z,w) = logp(x|z) + logp(z) + logp(w) + C
		# This is a proper scalar function
		logpw = 0
		for i in w:
			logpw += bnmodels.logpdfs.normal(w[i], 0, self.prior_sd).sum() # logp(w)
		
		return logpx, logpz, logpw

	# Confabulate hidden states 'z'
	def gen_xz(self, w, x, z):
		A = np.ones((1, self.n_batch))
		
		# Factors of X
		_z = {}
		sd = np.dot(np.exp(w['logsd']), A)
		for i in range(self.n_steps):
			if not z.has_key('eps'+str(i)):
				z['eps'+str(i)] = np.random.standard_normal(size=(self.n_z, self.n_batch))
			
			if i == 0:
				_z['z'+str(i)] = z['eps'+str(i)]
			else:
				_z['z'+str(i)] = z['eps'+str(i)] * sd
				_z['z'+str(i)] += np.tanh(np.dot(w['wz'], _z['z'+str(i-1)]) + np.dot(w['bz'], A))
			if not x.has_key('x'+str(i)):
				pi = 1/(1+np.exp(-np.dot(w['wx'], _z['z'+str(i)]) - np.dot(w['bx'], A))) # pi = p(X_i=1)
				x['x'+str(i)] = np.random.binomial(n=1,p=pi,size=pi.shape)
				
		return x, z, _z
	
	# Initial parameters
	def init_w(self, std = 1e-2):
		n_z, n_x = self.n_z, self.n_x
		w = {}
		w['wz'] = np.random.standard_normal(size=(n_z, n_z)) * std
		w['bz'] = np.random.standard_normal(size=(n_z, 1)) * std
		w['logsd'] = np.random.standard_normal(size=(n_z, 1)) * std
		w['wx'] = np.random.standard_normal(size=(n_x, n_z)) * std
		w['bx'] = np.random.standard_normal(size=(n_x, 1)) * std
		return w
	
	def variables(self):
		# Define parameters 'w'
		w = {}
		for i in ['wz','bz','logsd','wx','bx']:
			w[i] = T.dmatrix(i)
		
		# Define variables 'x' and 'z'
		z = {}
		x = {}
		for i in range(self.n_steps):
			z['eps'+str(i)] = T.dmatrix('eps'+str(i))
			x['x'+str(i)] = T.dmatrix('x'+str(i))
			
		return w, z, x
	
	