import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy as ap

import math, inspect

class MLBN(ap.BNModel):
	
    # n_basis is first hidden layer
	def __init__(self, n_hidden, n_basis, n_x, prior_sd=1, noMiddleEps=True, data='binary', nonlinear='softplus'):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_hidden = n_hidden
		self.n_basis = n_basis
		self.n_x = n_x
		self.prior_sd = prior_sd
		self.noMiddleEps = noMiddleEps
		self.data = data
		self.nonlinear = nonlinear
		super(MLBN, self).__init__()
	
	def factors(self, w, x, z, A):
		
		# Define symbolic program
		hidden = []
		hidden.append(z['eps0'])
		
		def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
		nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus}[self.nonlinear]

		for i in range(1, len(self.n_hidden)):
			h = nonlinear(T.dot(w['w'+str(i)], hidden[i-1]) + T.dot(w['b'+str(i)], A))
			if not self.noMiddleEps:
				h += z['eps'+str(i)] * T.dot(T.exp(w['logsd'+str(i)]), A)
			hidden.append(h)
		
		h_basis = T.tanh(T.dot(w['wbasis'], hidden[-1]) + T.dot(w['bbasis'], A))
		
		#p = 0.5 + 0.5 * T.tanh(T.dot(w['wout'], hidden[-1]) + T.dot(w['bout'], A)) # p = p(X=1)
		if self.data == 'binary':
			p = T.nnet.sigmoid(T.dot(w['wout'], h_basis) + T.dot(w['bout'], A))
			logpx = - T.nnet.binary_crossentropy(p, x['x'])
		elif self.data == 'gaussian'or self.data == 'sigmoidgaussian':
			x_mean = T.dot(w['wout'], h_basis) + T.dot(w['bout'], A)
			if self.data == 'sigmoidgaussian':
				x_mean = T.nnet.sigmoid(x_mean)
			x_logvar = T.dot(w['out_logvar_w'], h_basis) + T.dot(w['out_logvar_b'], A)
			logpx = ap.logpdfs.normal2(x['x'], x_mean, x_logvar)
		else: raise Exception("")

		# Note: logpx is a row vector (one element per sample)
		logpx = T.dot(np.ones((1, self.n_x)), logpx) # logpx = log p(x|z,w)
		
		# Note: logpz is a row vector (one element per sample)
		logpz = 0
		for i in z:
			logpz += ap.logpdfs.standard_normal(z[i]).sum(axis=0) # logp(z)
		
		# Note: logpw is a scalar
		def f_prior(_w):
			return ap.logpdfs.normal(_w, 0, self.prior_sd).sum()
		logpw = 0
		for i in range(1, len(self.n_hidden)):
			logpw += f_prior(w['w%i'%i])
		logpw += f_prior(w['wout'])
		if self.data == 'sigmoidgaussian' or self.data == 'gaussian':
			logpw += f_prior(w['out_logvar_w'])

		return logpw, logpx, logpz
	
	# Confabulate latent variables
	def gen_xz(self, w, x, z, n_batch):
		
		A = np.ones((1, n_batch))
		
		if not z.has_key('eps0'):
			z['eps0'] = np.random.standard_normal(size=(self.n_hidden[0], n_batch))
		
		_z = {'z0': z['eps0']}
		
		#def f_sigmoid(x): return 1e-3 + (1-2e-3) * 1/(1+np.exp(-x))
		def f_sigmoid(x): return 1./(1.+np.exp(-x))
		def f_softplus(x): return np.log(np.exp(x) + 1)# - np.log(2)
		nonlinear = {'tanh': np.tanh, 'sigmoid': f_sigmoid,'softplus': f_softplus}[self.nonlinear]

		for i in range(1, len(self.n_hidden)):
			mean = nonlinear(np.dot(w['w'+str(i)], _z['z'+str(i-1)]) + np.dot(w['b'+str(i)], A))
			_z['z'+str(i)] = mean
			
			if (not self.noMiddleEps):
				if not z.has_key('eps'+str(i)):
					z['eps'+str(i)] = np.random.standard_normal(size=(self.n_hidden[i], n_batch))
				sd = np.dot(np.exp(w['logsd'+str(i)]), A)
				_z['z'+str(i)] += z['eps'+str(i)] * sd
		
		h_basis = nonlinear(np.dot(w['wbasis'], _z['z'+str(len(self.n_hidden)-1)]) + np.dot(w['bbasis'], A))
		
		if self.data == 'binary':
			p = f_sigmoid(np.dot(w['wout'], h_basis) + np.dot(w['bout'], A))
			_z['x'] = p
			if not x.has_key('x'):
				x['x'] = np.random.binomial(n=1,p=p)
		elif self.data == 'sigmoidgaussian' or self.data == 'gaussian':
			x_mean = np.dot(w['wout'], h_basis) + np.dot(w['bout'], A)
			if self.data == 'sigmoidgaussian':
				x_mean = f_sigmoid(x_mean)
			x_logvar = np.dot(w['out_logvar_w'], h_basis) + np.dot(w['out_logvar_b'], A)
			_z['x'] = x_mean
			if not x.has_key('x'):
				x['x'] = np.random.normal(x_mean, np.exp(x_logvar/2))
		else: raise Exception("")
		
		return x, z, _z
	
	# Translate 'z' (hidden unit activations) to corresponding epsilons
	def z_to_eps(self, w, z, n_batch):
		eps = {}
		
		A = np.ones((1, n_batch))
				
		eps['eps0'] = z['z0']
		
		#def f_sigmoid(x): return 1e-3 + (1-2e-3) * 1/(1+np.exp(-x))
		def f_sigmoid(x): return 1./(1.+np.exp(-x))
		def f_softplus(x): return np.log(np.exp(x) + 1)# - np.log(2)
		nonlinear = {'tanh': np.tanh, 'sigmoid': f_sigmoid,'softplus': f_softplus}[self.nonlinear]

		for i in range(1, len(self.n_hidden)):
			mean = nonlinear(np.dot(w['w'+str(i)], z['z'+str(i-1)]) + np.dot(w['b'+str(i)], A))
			sd = np.dot(np.exp(w['logsd'+str(i)]), A)
			eps['eps'+str(i)] = (z['z'+str(i)] - mean) / sd
			
		return eps
	
	def variables(self):
		# Define parameters 'w'
		w = {}
		for i in range(1, len(self.n_hidden)):
			w['w'+str(i)] = T.dmatrix('w'+str(i))
			w['b'+str(i)] = T.dmatrix('b'+str(i))
			if not self.noMiddleEps:
				w['logsd'+str(i)] = T.dmatrix('logsd'+str(i))
		w['wbasis'] = T.dmatrix('wbasis')
		w['bbasis'] = T.dmatrix('bbasis')
		w['wout'] = T.dmatrix('wout')
		w['bout'] = T.dmatrix('bout')
		
		if self.data == 'sigmoidgaussian' or self.data == 'gaussian':
			w['out_logvar_w'] = T.dmatrix('out_logvar_w')
			w['out_logvar_b'] = T.dmatrix('out_logvar_b')
		
		# Define latent variables 'z'
		z = {'eps0': T.dmatrix('eps0')}
		if not self.noMiddleEps:
			for i in range(1, len(self.n_hidden)):
				z['eps'+str(i)] = T.dmatrix('eps'+str(i))
		
		# Define observed variables 'x'
		x = {}
		x['x'] = T.dmatrix('x')
		
		return w, x, z
	
	def init_w(self, std=1e-2):
		w = {}
		
		for i in range(1, len(self.n_hidden)):
			w['w'+str(i)] = np.random.normal(0, std, size=(self.n_hidden[i], self.n_hidden[i-1]))
			w['b'+str(i)] = np.random.normal(0, std, size=(self.n_hidden[i], 1))
			if not self.noMiddleEps:
				w['logsd'+str(i)] = np.random.normal(0, std, size=(self.n_hidden[i], 1))
		w['wbasis'] = np.random.normal(0, std, size=(self.n_basis, self.n_hidden[-1]))
		w['bbasis'] = np.random.normal(0, std, size=(self.n_basis, 1))
		w['wout'] = np.random.normal(0, std, size=(self.n_x, self.n_basis))
		w['bout'] = np.random.normal(0, std, size=(self.n_x, 1))
		if self.data == 'sigmoidgaussian' or self.data == 'gaussian':
			w['out_logvar_w'] = np.random.normal(0, std, size=(self.n_x, self.n_hidden[-1]))
			w['out_logvar_b'] = np.zeros((self.n_x, 1))

		return w
	