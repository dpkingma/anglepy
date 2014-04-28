import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy as ap

import math, inspect

'''
Deep Latent Variable Model (DLVM)
'''

class DLVM(ap.VAEModel):
	
	def __init__(self, n_x, n_hq, n_z, n_hp, nonlinear_q='softplus', nonlinear_p='softplus', type_px='bernoulli', type_qz='gaussianmarg', type_pz='gaussianmarg', prior_sd=1):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_x = n_x
		self.n_hq = n_hq
		self.n_z = n_z
		self.n_hp = n_hp
		self.nonlinear_q = nonlinear_q
		self.nonlinear_p = nonlinear_p
		self.type_px = type_px
		self.type_qz = type_qz
		self.type_pz = type_pz
		self.prior_sd = prior_sd
		
		if len(self.n_z) != len(self.n_hp) or len(self.n_z) != len(self.n_hq):
			raise Exception("Lengths are not equal.")
		
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

		# non-linearities
		def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
		def f_rectlin(x): return x*(x>0)
		nonlinear_q = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin}[self.nonlinear_q]
		nonlinear_p = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus, 'rectlin': f_rectlin}[self.nonlinear_p]
		
		# Forward pass through layers
		logpz = 0
		logqz = 0
		hq = []
		zs = []
		for i in range(len(self.n_z)):
			
			# Compute hidden-layer activations of encoder
			if i == 0:
				input = x['x']
			if i > 0:
				input = hq[i-1]
			
			q_w = w['q_w'+str(i)]			
			if nonlinear_q == 'rectlin':
				if i == 0:	B = np.ones((1, self.n_x))
				else:		B = np.ones((1, self.n_hq[i-1]))
				q_w = q_w / T.dot((T.dot(q_w**2, B.T)**0.5), B)
			
			hq.append(nonlinear_q(T.dot(q_w, input) + T.dot(w['q_b'+str(i)], A)))

			# Compute virtual sample
			if self.type_qz in ['gaussian', 'gaussianmarg']:
				q_mean = T.dot(w['q_mean_w'+str(i)], hq[i]) + T.dot(w['q_mean_b'+str(i)], A)
				q_logvar = T.dot(w['q_logvar_w'+str(i)], hq[i]) + T.dot(w['q_logvar_b'+str(i)], A)
				_z = q_mean + T.exp(0.5 * q_logvar) * z['eps'+str(i)]
				zs.append(_z)
			else: raise Exception()
			
			# log p(z) (prior of z)
			if self.type_pz == 'gaussian':
				logpz += ap.logpdfs.standard_normal(_z).sum(axis=0, keepdims=True)
			elif self.type_pz == 'gaussianmarg':
				logpz += -0.5 * (np.log(2 * np.pi) + (q_mean**2 + T.exp(q_logvar))).sum(axis=0, keepdims=True)
			elif self.type_pz == 'laplace':
				logpz += ap.logpdfs.standard_laplace(_z).sum(axis=0, keepdims=True)
			elif self.type_pz == 'studentt':
				logpz += ap.logpdfs.studentt(_z, T.dot(T.exp(w['logv'+str(i)]), A)).sum(axis=0, keepdims=True)
			else: raise Exception("Unknown type_pz")
			
			# loq q(z|x) (entropy of z)
			if self.type_qz == 'gaussian':
				logqz += ap.logpdfs.normal2(_z, q_mean, q_logvar).sum(axis=0, keepdims=True)
			elif self.type_qz == 'gaussianmarg':
				logqz += - 0.5 * (np.log(2 * np.pi) + 1 + q_logvar).sum(axis=0, keepdims=True)
			else: raise Exception('Unknown type_qz')

		# Compute decoder hidden layer activations
		hp = [0] * len(self.n_z)
		
		for j in range(len(self.n_z)):
			i = len(self.n_z)-j-1
			
			p_wz = w['p_wz'+str(i)]
			if self.nonlinear_p == 'rectlin':
				B = np.ones((1, self.n_z[i]))
				p_wz = p_wz / T.dot(T.dot(p_wz**2, B.T)**0.5, B)
				# TODO: also normalize p_w (not only p_wz)
			
			y = T.dot(p_wz, zs[i])
			if j > 0:
				y += T.dot(w['p_w'+str(i)], hp[i+1])
			y += T.dot(w['p_b'+str(i)], A)
			hp[i] = nonlinear_p(y)
		
		if self.type_px == 'bernoulli':
			p = T.nnet.sigmoid(T.dot(w['out_w'], hp[0]) + T.dot(w['out_b'], A))
			_logpx = - T.nnet.binary_crossentropy(p, x['x'])
		elif self.type_px == 'gaussian'or self.type_px == 'sigmoidgaussian':
			x_mean = T.dot(w['out_w'], hp[0]) + T.dot(w['out_b'], A)
			if self.type_px == 'sigmoidgaussian':
				x_mean = T.nnet.sigmoid(x_mean)
			x_logvar = T.dot(w['out_logvar_w'], hp[0]) + T.dot(w['out_logvar_b'], A)
			_logpx = ap.logpdfs.normal2(x['x'], x_mean, x_logvar)
		else: raise Exception('Unknown type_px')
			
		# Note: logpx is a row vector (one element per sample)
		logpx = T.dot(np.ones((1, self.n_x)), _logpx) # logpx = log p(x|z,w)
		
		# Note: logpw is a scalar
		def f_prior(_w, prior_sd=self.prior_sd):
			return ap.logpdfs.normal(_w, 0, prior_sd).sum()
		logpw = 0
		for i in range(len(self.n_hq)):
			logpw += f_prior(w['q_b'+str(i)])
			if self.nonlinear_q != 'rectlin':
				logpw += f_prior(w['q_w'+str(i)])
			if self.type_qz in ['gaussian','gaussianmarg']:
				logpw += f_prior(w['q_mean_b'+str(i)])
				logpw += f_prior(w['q_logvar_b'+str(i)])
				logpw += f_prior(w['q_mean_w'+str(i)])
				logpw += f_prior(w['q_logvar_w'+str(i)])
			if i < len(self.n_hq)-1:
				if self.nonlinear_p != 'rectlin':
					logpw += f_prior(w['p_w'+str(i)])
				logpw += f_prior(w['p_b'+str(i)])
			logpw += f_prior(w['p_wz'+str(i)])
		
		logpw += f_prior(w['out_w'])
		if self.type_px in ['sigmoidgaussian', 'gaussian']:
			logpw += f_prior(w['out_logvar_w'])
		if self.type_pz == 'studentt':
			for i in range(1, len(self.n_z)):
				logpw += f_prior(w['logv'+str(i)])
			
		return logpw, logpx, logpz, logqz, {}
	
	# TODO: separate 'eps' random variable (instead of being in 'z')
	
	# Generate epsilon from prior
	def gen_eps(self, n_batch):
		z = {'eps'+str(i): np.random.standard_normal(size=(self.n_z[i], n_batch)) for i in range(len(self.n_z))}
		return z
	
	# Generate variables
	def gen_xz(self, w, x, z, n_batch):

		# Require epsilon
		if not z.has_key('eps0'):
			_eps = self.gen_eps(n_batch)
			for i in _eps: z[i] = _eps[i]
			
		A = np.ones((1, n_batch))
		
		_z = {}

		#def f_sigmoid(x): return 1e-3 + (1-2e-3) * 1/(1+np.exp(-x))
		def f_sigmoid(x): return 1./(1.+np.exp(-x))
		def f_softplus(x): return np.log(np.exp(x) + 1)# - np.log(2)
		def f_rectlin(x): return x*(x>0)
		nonlinear_q = {'tanh': np.tanh, 'sigmoid': f_sigmoid,'softplus': f_softplus, 'rectlin': f_rectlin}[self.nonlinear_q]
		nonlinear_p = {'tanh': np.tanh, 'sigmoid': f_sigmoid,'softplus': f_softplus, 'rectlin': f_rectlin}[self.nonlinear_p]
		
		# If x['x'] was given: generate z ~ q(z|x)
		if x.has_key('x') and not z.has_key('z'):
			hq = []
			
			# Forward pass through layers
			for i in range(len(self.n_z)):
				
				# Compute hidden-layer activations of encoder
				if i == 0:
					input = x['x']
				if i > 0:
					input = hq[i-1]
					
				q_w = w['q_w'+str(i)]			
				if nonlinear_q == 'rectlin':
					if i == 0:	B = np.zeros((1, self.n_x))
					else:		B = np.zeros((1, self.n_hq[i-1]))
					q_w = q_w / np.dot((np.dot(q_w**2, B.T)**0.5), B)
	
				hq.append(nonlinear_q(np.dot(q_w, input) + np.dot(w['q_b'+str(i)], A)))
			
				# Compute virtual sample
				if self.type_qz in ['gaussian','gaussianmarg']:
					q_mean = np.dot(w['q_mean_w'+str(i)], hq[i]) + np.dot(w['q_mean_b'+str(i)], A)
					q_logvar = np.dot(w['q_logvar_w'+str(i)], hq[i]) + np.dot(w['q_logvar_b'+str(i)], A)
					_z['z_mean'+str(i)] = q_mean
					_z['z_logvar'+str(i)] = q_logvar
					z['z'+str(i)] = q_mean + np.exp(0.5 * q_logvar) * z['eps'+str(i)]
				else: raise Exception()
				
		elif not z.has_key('z'):
			for i in range(len(self.n_z)):
				if self.type_pz in ['gaussian','gaussianmarg']:
					z['z'+str(i)] = np.random.standard_normal(size=(self.n_z[i], n_batch))
				elif self.type_pz == 'laplace':
					z['z'+str(i)] = np.random.laplace(size=(self.n_z[i], n_batch))
				elif self.type_pz == 'studentt':
					z['z'+str(i)] = np.random.standard_t(np.dot(np.exp(w['logv']), A))
			
		# Compute decoder hidden layer activations
		hp = [0] * len(self.n_hp)
		for j in range(len(self.n_hp)):
			i = len(self.n_hp)-j-1
			
			p_wz = w['p_wz'+str(i)]
			if self.nonlinear_p == 'rectlin':
				B = np.zeros((1, self.n_z[i]))
				p_wz = p_wz / np.dot(np.dot(p_wz**2, B.T)**0.5, B)
				# TODO: also normalize p_w (not only p_wz)
			
			y = np.dot(p_wz, z['z'+str(i)])
			if j > 0:
				y += np.dot(w['p_w'+str(i)], hp[i+1])
			y += np.dot(w['p_b'+str(i)], A)
			hp[i] = nonlinear_p(y)
				
		if self.type_px == 'bernoulli':
			p = f_sigmoid(np.dot(w['out_w'], hp[0]) + np.dot(w['out_b'], A))
			_z['x'] = p
			if not x.has_key('x'):
				x['x'] = np.random.binomial(n=1,p=p)
		elif self.type_px == 'sigmoidgaussian' or self.type_px == 'gaussian':
			x_mean = np.dot(w['out_w'], hp[0]) + np.dot(w['out_b'], A)
			if self.type_px == 'sigmoidgaussian':
				x_mean = f_sigmoid(x_mean)
			x_logvar = np.dot(w['out_logvar_w'], hp[0]) + np.dot(w['out_logvar_b'], A)
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
		for i in range(len(self.n_hq)):
			w['q_w'+str(i)] = T.dmatrix('q_w'+str(i))
			w['q_b'+str(i)] = T.dmatrix('q_b'+str(i))
			w['q_mean_w'+str(i)] = T.dmatrix('q_mean_w'+str(i))
			w['q_mean_b'+str(i)] = T.dmatrix('q_mean_b'+str(i))
			if self.type_qz in ['gaussian','gaussianmarg']:
				w['q_logvar_w'+str(i)] = T.dmatrix('q_logvar_w'+str(i))
				w['q_logvar_b'+str(i)] = T.dmatrix('q_logvar_b'+str(i))
			if i < len(self.n_hq)-1:
				w['p_w'+str(i)] = T.dmatrix('p_w'+str(i))
			w['p_b'+str(i)] = T.dmatrix('p_b'+str(i))
			w['p_wz'+str(i)] = T.dmatrix('p_wz'+str(i))
		w['out_w'] = T.dmatrix('out_w')
		w['out_b'] = T.dmatrix('out_b')
		
		if self.type_px == 'sigmoidgaussian' or self.type_px == 'gaussian':
			w['out_logvar_w'] = T.dmatrix('out_logvar_w')
			w['out_logvar_b'] = T.dmatrix('out_logvar_b')
			
		if self.type_pz == 'studentt':
			for i in range(1, len(self.n_z)):
				w['logv'+str(i)] = T.dmatrix('logv'+str(i))

		# Define latent variables 'z'
		z = {}
		for i in range(len(self.n_z)):
			z['eps'+str(i)] = T.dmatrix('eps'+str(i))
		
		# Define observed variables 'x'
		x = {'x': T.dmatrix('x')}
		
		return w, x, z
	
	def init_w(self, std=1e-2):
		
		def rand(size):
			return np.random.normal(0, std, size=size)
		
		w = {}
		w['q_w0'] = rand((self.n_hq[0], self.n_x))
		w['q_b0'] = rand((self.n_hq[0], 1))
		for i in range(len(self.n_hq)):
			if i == 0:
				w['q_w'+str(i)] = rand((self.n_hq[i], self.n_x))
			else:
				w['q_w'+str(i)] = rand((self.n_hq[i], self.n_hq[i-1]))
			w['q_b'+str(i)] = rand((self.n_hq[i], 1))
			
			if self.type_qz in ['gaussian','gaussianmarg']:
				w['q_mean_w'+str(i)] = rand((self.n_z[i], self.n_hq[i]))
				w['q_mean_b'+str(i)] = rand((self.n_z[i], 1))
				w['q_logvar_w'+str(i)] = np.zeros((self.n_z[i], self.n_hq[i]))
				w['q_logvar_b'+str(i)] = np.zeros((self.n_z[i], 1))
			
			w['p_wz'+str(i)] = rand((self.n_hp[i], self.n_z[i]))
			w['p_b'+str(i)] = rand((self.n_hp[i], 1))
			if i < len(self.n_hq)-1:
				w['p_w'+str(i)] = rand((self.n_hp[i], self.n_hp[i+1]))
			
		w['out_w'] = rand((self.n_x, self.n_hp[0]))
		w['out_b'] = np.zeros((self.n_x, 1))
		if self.type_px in ['sigmoidgaussian', 'gaussian']:
			w['out_logvar_w'] = rand((self.n_x, self.n_hp[0]))
			w['out_logvar_b'] = np.zeros((self.n_x, 1))
		
		if self.type_pz == 'studentt':
			for i in range(1, len(self.n_z)):
				w['logv'+str(i)] = np.zeros((self.n_z[i], 1))
		
		return w
	