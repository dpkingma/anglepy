import theano
import theano.tensor as T
import numpy as np

import sys; sys.path.append('../../shared')
import anglepy as ap

# Write anglepy model
class MLP_Categorical(ap.BNModel):
	def __init__(self, n_units, prior_sd=1, nonlinearity='tanh'):
		self.n_units = n_units
		self.prior_sd = prior_sd
		self.nonlinearity = nonlinearity
		super(MLP_Categorical, self).__init__()
	
	def variables(self):
		w = {}
		for i in range(len(self.n_units)-1):
			w['w'+str(i)] = T.dmatrix()
			w['b'+str(i)] = T.dmatrix()
		
		x = {}
		x['x'] = T.dmatrix()
		x['t'] = T.dmatrix()
		
		z = {}
		return w, x, z
	
	def factors(self, w, x, z, A):
		# Define logp(w)
		logpw = 0
		for i in range(len(self.n_units)-1):
			logpw += ap.logpdfs.normal(w['w'+str(i)], 0, self.prior_sd).sum()
			logpw += ap.logpdfs.normal(w['b'+str(i)], 0, self.prior_sd).sum()
		
		if self.nonlinearity == 'tanh':
			f = T.tanh
		elif self.nonlinearity == 'sigmoid':
			f = T.nnet.sigmoid
		else:
			raise Exception("Unknown nonlinarity "+self.nonlinearity)
		
		# Define logp(x)
		hiddens  = [T.dot(w['w0'], x['x']) + T.dot(w['b0'], A)]
		for i in range(1, len(self.n_units)-1):
			hiddens.append(T.dot(w['w'+str(i)], f(hiddens[-1])) + T.dot(w['b'+str(i)], A))
		
		p = T.nnet.softmax(hiddens[-1].T).T
		logpx = - T.nnet.categorical_crossentropy(p.T, x['t'].T).T
		
		logpz = 0 * A
		return logpw, logpx, logpz, {'pt':(w.values() + [x['x'], A], p)}
		
	def gen_xz(self, w, x, z):
		if (not x.has_key('x') or x.has_key('t')):
			raise Exception('Not implemented')
		return x, z, {}
	
	def init_w(self, init_sd=1e-2):
		w = {}
		for i in range(len(self.n_units)-1):
			w['w'+str(i)] = init_sd * np.random.standard_normal((self.n_units[i+1], self.n_units[i]))
			w['b'+str(i)] = np.zeros((self.n_units[i+1], 1))
			
		return w

