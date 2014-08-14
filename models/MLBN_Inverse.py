import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.BNModel as BNModel
import anglepy.logpdfs as theano_funcs
import math, inspect
import anglepy.ndict as ndict

class MLBN_Inverse(BNModel):
	def __init__(self, n_units, prior_sd=1, nonlinear='softplus'):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_units = n_units
		self.prior_sd = prior_sd
		self.logvar_factor = 1e-1
		self.logvar_const = 0
		self.logmeanb_factor = 1
		self.nonlinear = nonlinear
		super(MLBN_Inverse, self).__init__('ignore')
		
	def factors(self, w, x, z, A):
		
		# Define symbolic program
		hidden = []
		hidden.append(x['x'])
		
		def f_softplus(x): return T.log(T.exp(x) + 1)# - np.log(2)
		nonlinear = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softplus': f_softplus}[self.nonlinear]
		
		for i in range(1, len(self.n_units)-1):
			hidden.append(nonlinear(T.dot(w['w_%i'%i], hidden[i-1]) + T.dot(w['b_'+str(i)], A)))
		
		mean = T.dot(w['w_mean'], hidden[-1]) + self.logmeanb_factor * T.dot(w['b_mean'], A)
		logvar = self.logvar_const + self.logvar_factor * T.dot(w['w_logvar'], hidden[-1]) + T.dot(w['b_logvar'], A)
		
		logpz = theano_funcs.normal2(z['eps0'], mean, logvar)
		
		# Note: logpx is a row vector (one element per sample)
		logpz = T.dot(np.ones((1, self.n_units[-1])), logpz) # logpx = log p(x|z,w)
		
		# Note: logpz is a row vector (one element per sample)
		logpx = T.zeros_like(logpz)
		
		# Note: logpw is a scalar
		def f_prior(_w):
			return theano_funcs.normal(_w, 0, self.prior_sd).sum()
		logpw = 0
		for i in range(1, len(self.n_units)-1):
			logpw += f_prior(w['w_'+str(i)])
		logpw += f_prior(w['w_mean'])
		logpw += f_prior(w['w_logvar'])
		
		return logpw, logpx, logpz
	
	# Confabulate hidden states 'z'
	def gen_xz(self, w, x, z, n_batch):
		A = np.ones((1, n_batch))
		
		if not x.has_key('x'):
			x['x'] = np.random.binomial(1, 0.5, size=(self.n_units[0], n_batch))
		
		def f_sigmoid(x): return 1./(1.+np.exp(-x))
		def f_softplus(x): return np.log(np.exp(x) + 1)# - np.log(2)
		nonlinear = {'tanh': np.tanh, 'sigmoid': f_sigmoid,'softplus': f_softplus}[self.nonlinear]

		hidden = []
		hidden.append(x['x'])
		for i in range(1, len(self.n_units)-1):
			hidden.append(nonlinear(np.dot(w['w_%i'%i], hidden[i-1]) + np.dot(w['b_%i'%i], A)))
		
		mean = np.dot(w['w_mean'], hidden[-1]) + self.logmeanb_factor * np.dot(w['b_mean'], A)
		logvar = self.logvar_const + self.logvar_factor * np.dot(w['w_logvar'], hidden[-1]) + np.dot(w['b_logvar'], A)
		
		if not z.has_key('eps0'):
			z['eps0'] = np.random.normal(mean, np.exp(logvar/2))
		
		if ndict.hasNaN(z):
			raise Exception("NaN detected")
		
		_z = {'mean':mean, 'logvar':logvar}
		return x, z, _z
		
	def variables(self):
		# Define parameters 'w'
		w = {}
		for i in range(1, len(self.n_units)-1):
			w['w_'+str(i)] = T.dmatrix('w_'+str(i))
			w['b_'+str(i)] = T.dmatrix('b_'+str(i))
		w['w_mean'] = T.dmatrix('w_mean')
		w['b_mean'] = T.dmatrix('b_mean')
		w['w_logvar'] = T.dmatrix('w_logvar')
		w['b_logvar'] = T.dmatrix('b_logvar')
		
		# Define observed variables 'x'
		x = {}
		x['x'] = T.dmatrix('x')
		
		# Latent variables
		z = {}
		z['eps0'] = T.dmatrix('eps0')
		
		return w, x, z
	
	def init_w(self, std=1e-2):
		w = {}
		for i in range(1, len(self.n_units)-1):
			w['w_'+str(i)] = np.random.normal(0, std, size=(self.n_units[i], self.n_units[i-1]))
			w['b_'+str(i)] = np.random.normal(0, std, size=(self.n_units[i], 1))
		w['w_mean'] = np.random.normal(0, std, size=(self.n_units[-1], self.n_units[-2]))
		w['b_mean'] = np.random.normal(0, std, size=(self.n_units[-1], 1))
		w['w_logvar'] = np.zeros((self.n_units[-1], self.n_units[-2]))
		w['b_logvar'] = np.zeros((self.n_units[-1], 1))
		return w
	