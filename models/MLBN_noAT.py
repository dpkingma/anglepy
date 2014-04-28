import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.BNModel as BNModel
import anglepy.logpdfs as logpdfs
import math, inspect
import anglepy as ap

class MLBN_CP(ap.BNModel):

	def __init__(self, n_hidden, n_basis, n_output, prior_sd=1):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_hidden = n_hidden
		self.n_basis = n_basis
		self.n_output = n_output
		self.prior_sd = prior_sd
		super(MLBN_CP, self).__init__()


	#A = np.ones((1, n_batch))
	
	def factors(self, w, x, z, A):
		
		# Define logp(z0)
		logpz = logpdfs.standard_normal(z['z0']).sum(axis=0)
		
		# Define logp(z_{n+1}|z_n)
		z_prev = z['z0']
		for i in range(1, len(self.n_hidden)):
			mean = T.tanh(T.dot(w['w'+str(i)], z_prev) + T.dot(w['b'+str(i)], A))
			sd = T.dot(T.exp(w['logsd'+str(i)]), A)
			logpz += logpdfs.normal(z['z'+str(i)], mean, sd).sum(axis=0)
			z_prev = z['z'+str(i)]
		
		z_basis = T.tanh(T.dot(w['wbasis'], z_prev) + T.dot(w['bbasis'], A))
		
		# logp(x|z_{last})	
		p = 1e-3 + (1-2e-3) * T.nnet.sigmoid(T.dot(w['wout'], z_basis) + T.dot(w['bout'], A)) # p = p(X=1)
		logpx = - T.nnet.binary_crossentropy(p, x['x'])
		logpx = T.dot(np.ones((1, self.n_output)), logpx)
		
		# joint() = logp(x,z,w) = logp(x,z|w) + logp(w) = logpxz + logp(w) + C
		# This is a proper scalar function
		logpw = 0
		for i in w:
			logpw += logpdfs.normal(w[i], 0, self.prior_sd).sum() # logp(w)
		
		return logpw, logpx, logpz, {}
		
	def variables(self):
		# Define parameters 'w'
		w = {}
		for i in range(1, len(self.n_hidden)):
			w['w'+str(i)] = T.dmatrix('w'+str(i))
			w['b'+str(i)] = T.dmatrix('b'+str(i))
			w['logsd'+str(i)] = T.dmatrix('logsd'+str(i))
		w['wbasis'] = T.dmatrix('wbasis')
		w['bbasis'] = T.dmatrix('bbasis')
		w['wout'] = T.dmatrix('wout')
		w['bout'] = T.dmatrix('bout')
		
		# Define latent variables 'z'
		z = {}
		for i in range(len(self.n_hidden)):
			z['z'+str(i)] = T.dmatrix('z'+str(i))
		
		# Define observed variables 'x'
		x = {}
		x['x'] = T.dmatrix('x')
		
		return w, x, z
	
	# Confabulate latent variables
	def gen_xz(self, w, x, z, n_batch):
		A = np.ones((1, n_batch))
		
		size = (self.n_hidden[0], n_batch)
		_z = {}
		
		if not z.has_key('z0'):
			z['z0'] = np.random.standard_normal(size=size)
		for i in range(1, len(self.n_hidden)):
			if not z.has_key('z'+str(i)):
				sd = np.dot(np.absolute(w['logsd'+str(i)]), A)
				mean = np.tanh(np.dot(w['w'+str(i)], z['z'+str(i-1)]) + np.dot(w['b'+str(i)], A))
				size = (self.n_hidden[i], n_batch)
				z['z'+str(i)] = np.random.normal(mean, sd, size=size)
		
		if not x.has_key('x'):
			h_basis = np.tanh(np.dot(w['wbasis'], z['z'+str(len(self.n_hidden)-1)]) + np.dot(w['bbasis'], A))
			p = 1e-3 + (1-2e-3) * 1./(1 + np.exp(np.dot(w['wout'], h_basis) + np.dot(w['bout'], A)))
			_z['x'] = p
			x['x'] = np.random.binomial(n=1,p=p)
		
		return x, z, _z
	
	def init_w(self, std=1e-2):
		n_hidden = self.n_hidden
		n_basis = self.n_basis
		n_output = self.n_output
		w = {}
		for i in range(1, len(n_hidden)):
			w['w'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], n_hidden[i-1]))
			w['b'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], 1))
			w['logsd'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], 1))
		w['wbasis'] = np.random.normal(0, std, size=(n_basis, n_hidden[-1]))
		w['bbasis'] = np.random.normal(0, std, size=(n_basis, 1))
		w['wout'] = np.random.normal(0, std, size=(n_output, n_basis))
		w['bout'] = np.random.normal(0, std, size=(n_output, 1))
		return w
	