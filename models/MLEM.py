import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy as ap

import math, inspect

'''
Fully connected Multi-layer Euler Machine with tanh(.) activations
'''

class DLVM(ap.VAEModel):
	
	def __init__(self, n_x, n_hidden_q, n_z, n_hidden_p, nonlinear_q='tanh', nonlinear_p='tanh', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_x = n_x
		self.n_hidden_q = n_hidden_q
		self.n_z = n_z
		self.n_hidden_p = n_hidden_p
		self.nonlinear_q = nonlinear_q
		self.nonlinear_p = nonlinear_p
		self.type_px = type_px
		self.type_qz = type_qz
		self.type_pz = type_pz
		self.prior_sd = prior_sd
		
		super(DLVM, self).__init__()
	
	def factors(self, w, x, z, A):
			
		'''
		z['eps'] is the independent epsilons (Gaussian with unit variance)
		x['x'] is the data
		
		The names of list z[...] may be confusing here: the latent variable z is not included in the list z[...],
		but implicitely computed from epsilon and parameters in w.

		z is computed with g(.) from eps and variational parameters
		let logpx be the generative model density: log p(x|z) where z=g(.)
		let logpz be the prior of Z plus the entropy of q(z|x): logp(z) + H_q(z|x)
		So the lower bound L(x) = logpx + logpz
		
		let logpw be the (prior) density of the weights
		'''
		
		# Compute q(z|x)
		hidden_q = [x['x']]
		
		def f_softrect(x): return T.log(T.exp(x) + 1)# - np.log(2)
		nonlinear_q = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softrect': f_softrect}[self.nonlinear_q]
		nonlinear_p = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softrect': f_softrect}[self.nonlinear_p]
		
		for i in range(len(self.n_hidden_q)):
			hidden_q.append(nonlinear_q(T.dot(w['q_w'+str(i)], hidden_q[-1]) + T.dot(w['q_b'+str(i)], A)))
		
		q_mean = T.dot(w['q_mean_w'], hidden_q[-1]) + T.dot(w['q_mean_b'], A)
		if self.type_qz == 'gaussian' or self.type_qz == 'gaussianmarg':
			q_logvar = T.dot(w['q_logvar_w'], hidden_q[-1]) + T.dot(w['q_logvar_b'], A)
		else: raise Exception()
		
		# Compute virtual sample
		_z = q_mean + T.exp(0.5 * q_logvar) * z['eps']
		
		# Compute log p(x|z)
		hidden_p = [_z]
		for i in range(len(self.n_hidden_p)):
			hidden_p.append(nonlinear_p(T.dot(w['p_w'+str(i)], hidden_p[-1]) + T.dot(w['p_b'+str(i)], A)))
		
		if self.type_px == 'bernoulli':
			p = T.nnet.sigmoid(T.dot(w['out_w'], hidden_p[-1]) + T.dot(w['out_b'], A))
			_logpx = - T.nnet.binary_crossentropy(p, x['x'])
		elif self.type_px == 'gaussian'or self.type_px == 'sigmoidgaussian':
			x_mean = T.dot(w['out_w'], hidden_p[-1]) + T.dot(w['out_b'], A)
			if self.type_px == 'sigmoidgaussian':
				x_mean = T.nnet.sigmoid(x_mean)
			x_logvar = T.dot(w['out_logvar_w'], hidden_p[-1]) + T.dot(w['out_logvar_b'], A)
			_logpx = ap.logpdfs.normal2(x['x'], x_mean, x_logvar)
		else: raise Exception("")
			
		# Note: logpx is a row vector (one element per sample)
		logpx = T.dot(np.ones((1, self.n_x)), _logpx) # logpx = log p(x|z,w)
		
		# log p(z) (prior of z)
		if self.type_pz == 'gaussianmarg':
			logpz = -0.5 * (np.log(2 * np.pi) + (q_mean**2 + T.exp(q_logvar))).sum(axis=0, keepdims=True)
		if self.type_pz == 'gaussian':
			logpz = ap.logpdfs.standard_normal(_z).sum(axis=0, keepdims=True)
		elif self.type_pz == 'laplace':
			logpz = ap.logpdfs.standard_laplace(_z).sum(axis=0, keepdims=True)
		elif self.type_pz == 'studentt':
			logpz = ap.logpdfs.studentt(_z, T.dot(T.exp(w['logv']), A)).sum(axis=0, keepdims=True)
		else:
			raise Exception("Unknown type_pz")
		
		# loq q(z|x) (entropy of z)
		if self.type_qz == 'gaussianmarg':
			logqz = - 0.5 * (np.log(2 * np.pi) + 1 + q_logvar).sum(axis=0, keepdims=True)
		elif self.type_qz == 'gaussian':
			logqz = ap.logpdfs.normal2(_z, q_mean, q_logvar).sum(axis=0, keepdims=True)
		else: raise Exception()
		
		# Note: logpw is a scalar
		def f_prior(_w, prior_sd=self.prior_sd):
			return ap.logpdfs.normal(_w, 0, prior_sd).sum()
		logpw = 0
		for i in range(len(self.n_hidden_q)):
			logpw += f_prior(w['q_w'+str(i)])
		logpw += f_prior(w['q_mean_w'])
		if self.type_qz in ['gaussian','gaussianmarg']:
			logpw += f_prior(w['q_logvar_w'])
		for i in range(len(self.n_hidden_p)):
			logpw += f_prior(w['p_w'+str(i)])
		logpw += f_prior(w['out_w'])
		if self.type_px in ['sigmoidgaussian', 'gaussian']:
			logpw += f_prior(w['out_logvar_w'])
		if self.type_pz == 'studentt':
			logpw += f_prior(w['logv'])
			
		return logpw, logpx, logpz, logqz, {}
	
	# Generate epsilon from prior
	def gen_eps(self, n_batch):
		z = {'eps': np.random.standard_normal(size=(self.n_z, n_batch))}
		return z
	
	# Generate variables
	def gen_xz(self, w, x, z, n_batch):

		# Require epsilon
		if not z.has_key('eps'):
			z['eps'] = self.gen_eps(n_batch)['eps']
			#raise Exception('epsilon required') 
		
		A = np.ones((1, n_batch))
		
		_z = {}

		#def f_sigmoid(x): return 1e-3 + (1-2e-3) * 1/(1+np.exp(-x))
		def f_sigmoid(x): return 1./(1.+np.exp(-x))
		def f_softrect(x): return np.log(np.exp(x) + 1)# - np.log(2)
		nonlinear_q = {'tanh': np.tanh, 'sigmoid': f_sigmoid,'softrect': f_softrect}[self.nonlinear_q]
		nonlinear_p = {'tanh': np.tanh, 'sigmoid': f_sigmoid,'softrect': f_softrect}[self.nonlinear_p]
		
		# If x['x'] was given: generate z ~ q(z|x)
		if x.has_key('x') and not z.has_key('z'):

			hidden_q = [x['x']]
			for i in range(len(self.n_hidden_q)):
				hidden_q.append(nonlinear_q(np.dot(w['q_w'+str(i)], hidden_q[-1]) + np.dot(w['q_b'+str(i)], A)))
			
			q_mean = np.dot(w['q_mean_w'], hidden_q[-1]) + np.dot(w['q_mean_b'], A)
			if self.type_qz in ['gaussian','gaussianmarg']:
				q_logvar = np.dot(w['q_logvar_w'], hidden_q[-1]) + np.dot(w['q_logvar_b'], A)
			else: raise Exception()
			
			_z['mean'] = q_mean
			_z['logvar'] = q_logvar
			
			z['z'] = q_mean + np.exp(0.5 * q_logvar) * z['eps']
			
		elif not z.has_key('z'):
			if self.type_pz in ['gaussian','gaussianmarg']:
				z['z'] = np.random.standard_normal(size=(self.n_z, n_batch))
			elif self.type_pz == 'laplace':
				z['z'] = np.random.laplace(size=(self.n_z, n_batch))
			elif self.type_pz == 'studentt':
				z['z'] = np.random.standard_t(np.dot(np.exp(w['logv']), A))
		
		# Generate from p(x|z)
		hidden_p = [z['z']]
		for i in range(len(self.n_hidden_p)):
			hidden_p.append(nonlinear_p(np.dot(w['p_w'+str(i)], hidden_p[-1]) + np.dot(w['p_b'+str(i)], A)))
		
		if self.type_px == 'bernoulli':
			p = f_sigmoid(np.dot(w['out_w'], hidden_p[-1]) + np.dot(w['out_b'], A))
			_z['x'] = p
			if not x.has_key('x'):
				x['x'] = np.random.binomial(n=1,p=p)
		elif self.type_px == 'sigmoidgaussian' or self.type_px == 'gaussian':
			x_mean = np.dot(w['out_w'], hidden_p[-1]) + np.dot(w['out_b'], A)
			if self.type_px == 'sigmoidgaussian':
				x_mean = f_sigmoid(x_mean)
			x_logvar = np.dot(w['out_logvar_w'], hidden_p[-1]) + np.dot(w['out_logvar_b'], A)
			_z['x'] = x_mean
			if not x.has_key('x'):
				x['x'] = np.random.normal(x_mean, np.exp(x_logvar/2))
				if self.type_px == 'sigmoidgaussian':
					x['x'] = np.maximum(np.zeros(x['x'].shape), x['x'])
					x['x'] = np.minimum(np.ones(x['x'].shape), x['x'])
		
		else: raise Exception("")
		
		return x, z, _z
	
	def variables(self):
		
		# Define parameters 'w'
		w = {}
		for i in range(len(self.n_hidden_q)):
			w['q_w'+str(i)] = T.dmatrix('q_w'+str(i))
			w['q_b'+str(i)] = T.dmatrix('q_b'+str(i))
		w['q_mean_w'] = T.dmatrix('q_mean_w')
		w['q_mean_b'] = T.dmatrix('q_mean_b')
		if self.type_qz in ['gaussian','gaussianmarg']:
			w['q_logvar_w'] = T.dmatrix('q_logvar_w')
		w['q_logvar_b'] = T.dmatrix('q_logvar_b')
		for i in range(len(self.n_hidden_p)):
			w['p_w'+str(i)] = T.dmatrix('p_w'+str(i))
			w['p_b'+str(i)] = T.dmatrix('p_b'+str(i))
		w['out_w'] = T.dmatrix('out_w')
		w['out_b'] = T.dmatrix('out_b')
		
		if self.type_px == 'sigmoidgaussian' or self.type_px == 'gaussian':
			w['out_logvar_w'] = T.dmatrix('out_logvar_w')
			w['out_logvar_b'] = T.dmatrix('out_logvar_b')
			
		if self.type_pz == 'studentt':
			w['logv'] = T.dmatrix('logv')

		# Define latent variables 'z'
		z = {'eps': T.dmatrix('eps')}
		
		# Define observed variables 'x'
		x = {'x': T.dmatrix('x')}
		
		
		return w, x, z
	
	def init_w(self, std=1e-2):
		
		def rand(size):
			return np.random.normal(0, std, size=size)
		
		w = {}
		w['q_w0'] = rand((self.n_hidden_q[0], self.n_x))
		w['q_b0'] = rand((self.n_hidden_q[0], 1))
		for i in range(1, len(self.n_hidden_q)):
			w['q_w'+str(i)] = rand((self.n_hidden_q[i], self.n_hidden_q[i-1]))
			w['q_b'+str(i)] = rand((self.n_hidden_q[i], 1))
		
		w['q_mean_w'] = rand((self.n_z, self.n_hidden_q[-1]))
		w['q_mean_b'] = rand((self.n_z, 1))
		if self.type_qz in ['gaussian','gaussianmarg']:
			w['q_logvar_w'] = np.zeros((self.n_z, self.n_hidden_q[-1]))
		w['q_logvar_b'] = np.zeros((self.n_z, 1))
		
		if len(self.n_hidden_p) > 0:
			w['p_w0'] = rand((self.n_hidden_p[0], self.n_z))
			w['p_b0'] = rand((self.n_hidden_p[0], 1))
			for i in range(1, len(self.n_hidden_p)):
				w['p_w'+str(i)] = rand((self.n_hidden_p[i], self.n_hidden_p[i-1]))
				w['p_b'+str(i)] = rand((self.n_hidden_p[i], 1))
			w['out_w'] = rand((self.n_x, self.n_hidden_p[-1]))
			w['out_b'] = np.zeros((self.n_x, 1))
			if self.type_px in ['sigmoidgaussian', 'gaussian']:
				w['out_logvar_w'] = rand((self.n_x, self.n_hidden_p[-1]))
				w['out_logvar_b'] = np.zeros((self.n_x, 1))
		else:
			w['out_w'] = rand((self.n_x, self.n_z))
			w['out_b'] = np.zeros((self.n_x, 1))
			if self.type_px in ['sigmoidgaussian', 'gaussian']:
				w['out_logvar_w'] = rand((self.n_x, self.n_z))
				w['out_logvar_b'] = np.zeros((self.n_x, 1))
		
		if self.type_pz == 'studentt':
			w['logv'] = np.zeros((self.n_z, 1))
		
		return w
	