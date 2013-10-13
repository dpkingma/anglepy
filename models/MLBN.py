import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.BNModel as BNModel
import anglepy.logpdfs as theano_funcs
import math, inspect

class Model(BNModel.BNModel):
	def __init__(self, n_hidden, n_output, n_batch, prior_sd=1, noMiddleEps=False):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_hidden = n_hidden
		self.n_output = n_output
		self.n_batch = n_batch
		self.prior_sd = prior_sd
		self.noMiddleEps = noMiddleEps
		super(Model, self).__init__(n_batch)
	
	def factors(self, w, z, x):
		
		A = np.ones((1, self.n_batch))
	
		# Define symbolic program
		hidden = []
		hidden.append(z['eps0'])
		
		for i in range(1, len(self.n_hidden)):
			h = T.tanh(T.dot(w['w%d'%i], hidden[i-1]) + T.dot(w['b'+str(i)], A))
			if not self.noMiddleEps:
				h += z['eps'+str(i)] * T.dot(T.abs_(w['logsd'+str(i)]), A)
			hidden.append(h)
		
		#p = 0.5 + 0.5 * T.tanh(T.dot(w['wout'], hidden[-1]) + T.dot(w['bout'], A)) # p = p(X=1)
		p = T.nnet.sigmoid(T.dot(w['wout'], hidden[-1]) + T.dot(w['bout'], A))
		
		logpx = - T.nnet.binary_crossentropy(p, x['x'])
		
		# Note: logpx is a row vector (one element per sample)
		logpx = T.dot(np.ones((1, self.n_output)), logpx) # logpx = log p(x|z,w)
		
		# Note: logpz is a row vector (one element per sample)
		logpz = 0
		for i in z:
			logpz += theano_funcs.standard_normal(z[i]).sum(axis=0) # logp(z)
		
		# Note: logpw is a scalar
		def f_prior(_w):
			return theano_funcs.normal(_w, 0, self.prior_sd).sum()
		logpw = 0
		for i in range(1, len(self.n_hidden)):
			logpw += f_prior(w['w%i'%i])
		logpw += f_prior(w['wout'])
		
		return logpx, logpz, logpw
	
	# Confabulate latent variables
	def gen_xz(self, w, x, z):
		A = np.ones((1, self.n_batch))
		
		if not z.has_key('eps0'):
			z['eps0'] = np.random.standard_normal(size=(self.n_hidden[0], self.n_batch))
										
		_z = {'z0': z['eps0']}
		
		for i in range(1, len(self.n_hidden)):
			mean = np.tanh(np.dot(w['w'+str(i)], _z['z'+str(i-1)]) + np.dot(w['b'+str(i)], A))
			_z['z'+str(i)] = mean
			
			if (not self.noMiddleEps):
				if not z.has_key('eps'+str(i)):
					z['eps'+str(i)] = np.random.standard_normal(size=(self.n_hidden[i], self.n_batch))
				sd = np.dot(np.absolute(w['logsd'+str(i)]), A)
				_z['z'+str(i)] += z['eps'+str(i)] * sd
		
		if not x.has_key('x'):
			p = 1e-3 + (1-2e-3) * 1/(1+np.exp(-(np.dot(w['wout'], _z['z'+str(len(self.n_hidden)-1)]) + np.dot(w['bout'], A))))
			_z['x'] = p
			x['x'] = np.random.binomial(n=1,p=p)
		
		return x, z, _z

	def variables(self):
		# Define parameters 'w'
		w = {}
		for i in range(1, len(self.n_hidden)):
			w['w'+str(i)] = T.dmatrix('w'+str(i))
			w['b'+str(i)] = T.dmatrix('b'+str(i))
			if not self.noMiddleEps:
				w['logsd'+str(i)] = T.dmatrix('logsd'+str(i))
		w['wout'] = T.dmatrix('wout')
		w['bout'] = T.dmatrix('bout')
		
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
		w['wout'] = np.random.normal(0, std, size=(self.n_output, self.n_hidden[-1]))
		w['bout'] = np.random.normal(0, std, size=(self.n_output, 1))
		return w
	