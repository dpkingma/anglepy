import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.BNModel as BNModel
import anglepy.logpdfs as logpdfs
import math, inspect
import anglepy as ap
from theano.tensor.shared_randomstreams import RandomStreams

class DBN_noAT(BNModel):
	
	def __init__(self, n_z, n_x, n_steps, prior_sd=0.1, data='binary'):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_z, self.n_x, self.n_steps, self.prior_sd = n_z, n_x, n_steps, prior_sd
		self.data = data
		theano_warning = 'raise'
		if n_steps == 1: theano_warning = 'warn'

		super(DBN_noAT, self).__init__(theano_warning)
	
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
		
		# Factors of X and Z
		logpx = 0
		logpz = 0
		sd = T.dot(T.exp(w['logsd']), A)
		for i in range(self.n_steps):
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
			logpw += logpdfs.normal(w[i], 0, self.prior_sd).sum() # logp(w)
		
		return logpw, logpx, logpz, {}
	
	# Confabulate latent variables 'x' and 'z'
	def gen_xz(self, w, x, z, n_batch):
		
		A = np.ones((1, n_batch))

		sd = np.dot(np.exp(w['logsd']), A)
		for i in range(self.n_steps):
			if not z.has_key('z'+str(i)):
				if i == 0:
					z['z'+str(i)] = np.random.standard_normal(size=(self.n_z, n_batch))
				else:
					mean = np.tanh(np.dot(w['wz'], z['z'+str(i-1)]) + np.dot(w['bz'], A))
					z['z'+str(i)] = mean + sd * np.random.standard_normal(size=(self.n_z, n_batch))
			
			if not x.has_key('x'+str(i)):
				if self.data == 'binary':
					pi = 1/(1+np.exp(-np.dot(w['wx'], z['z'+str(i)]) - np.dot(w['bx'], A))) # pi = p(X_i=1)
					x['x'+str(i)] = np.random.binomial(n=1,p=pi,size=pi.shape)
				elif self.data == 'gaussian':
					x_mean = np.dot(w['wx'], z['z'+str(i)]) + np.dot(w['bx'], A)
					x_logvar = np.dot(2*w['logsdx'], A)
					x['x'+str(i)] = np.random.normal(x_mean, np.exp(x_logvar/2))
				else: raise Exception()
			
		return x, z, {}

	# Initial parameters
	def init_w(self, std = 1e-2):
		n_z, n_x = self.n_z, self.n_x
		w = {}
		w['wz'] = np.random.normal(0, std, size=(n_z, n_z))
		w['bz'] = np.random.normal(0, std, size=(n_z, 1))
		w['logsd'] = np.random.normal(0, std, size=(n_z, 1))
		w['wx'] = np.random.normal(0, std, size=(n_x, n_z))
		w['bx'] = np.random.normal(0, std, size=(n_x, 1))
		if self.data == 'gaussian':
			w['logsdx'] = np.random.standard_normal(size=(n_z, 1)) * std
		return w
	
	def variables(self):
		# Define parameters 'w'
		w = {}
		w['wz'] = T.dmatrix('wz')
		w['bz'] = T.dmatrix('bz')
		w['logsd'] = T.dmatrix('logsd')
		w['wx'] = T.dmatrix('wx')
		w['bx'] = T.dmatrix('bx')
		w['logsdx'] = T.dmatrix('logsdx')
		
		# Define variables 'x' and 'z'
		z = {}
		x = {}
		for i in range(self.n_steps):
			z['z'+str(i)] = T.dmatrix('z'+str(i))
			x['x'+str(i)] = T.dmatrix('x'+str(i))
		
		return w, x, z
	