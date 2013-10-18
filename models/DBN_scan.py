import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.logpdfs
import anglepy.BNModel as BNModel
import math, inspect
import anglepy.ndict as ndict
import sys

'''
	Bugfix: inserted in line 252 of
	/Library/Frameworks/EPD64.framework/Versions/7.3/lib/python2.7/site-packages/theano/scan_module/scan_op.py
	=> return rval
'''

class DBN_scan(BNModel):
	
	def __init__(self, n_z, n_x, n_steps, prior_sd=0.1):
		self.constr = (__name__, inspect.stack()[0][3], locals())
		self.n_z, self.n_x, self.n_steps = n_z, n_x, n_steps
		self.prior_sd = prior_sd
		
		theano_warning = 'raise'
		if n_steps == 1: theano_warning = 'warn'
		
		super(DBN_scan, self).__init__(theano_warning)

	def variables(self):
		# Define parameters 'w'
		w = {}
		for i in ['wz','bz','logsd','wx','bx']:
			w[i] = T.dmatrix(i)
		
		# Define variables 'x' and 'z'
		z = {'eps':T.dtensor3('eps')}
		x = {'x':T.dtensor3('x')}
		
		return w, x, z
	
	def factors(self, w, x, z, A):
		
		def pr(str, _x):
			return theano.printing.Print(str)(_x)
		
		#A = T.ones_like(x['x'][0,0:1]) #np.ones((1, self.n_batch))
		
		# Removed in refactoring
		#A = np.ones((1, self.n_batch))
		#A = T.unbroadcast(T.constant(A), 0) 
		
		# Init sequences
		
		def f_xi(_z, _x):
			pi = T.nnet.sigmoid(T.dot(w['wx'], _z) + T.dot(w['bx'], A)) # pi = p(X_i=1)
			logpxi = - T.nnet.binary_crossentropy(pi, _x)
			return logpxi.sum(axis=0, keepdims=True)
		
		# Scan op
		def f_step(eps_t, x_t, _z_prev, logpx_prev):
			z_t = eps_t * T.dot(T.exp(w['logsd']), A)
			z_t += T.tanh(T.dot(w['wz'], _z_prev) + T.dot(w['bz'], A))
			logpx_t = f_xi(z_t, x_t)
			return [z_t+0, logpx_t] #+0 prevents bug
		
		# Compute variables of first timestep
		_z_0 = z['eps'][0]
		logpx_0 = f_xi(_z_0, x['x'][0])
		
		# Perform loop
		[_zs, logpxs], _ = theano.scan(
			fn = f_step, \
			sequences = [z['eps'], x['x']], \
			outputs_info = [_z_0, logpx_0],
			n_steps = self.n_steps
		)
		
		logpx = logpxs.sum(axis=0).sum(axis=0, keepdims=True)
		
		# Factors of Z
		logpz = anglepy.logpdfs.standard_normal(z['eps']).sum(axis=0).sum(axis=0, keepdims=True)
		
		logpw = 0
		for i in w:
			logpw += anglepy.logpdfs.normal(w[i], 0, self.prior_sd).sum() # logp(w)
		
		return logpw, logpx, logpz, {}
	
	# Numpy <-> Theano var conversion
	def xz_to_theano(self, x, z):
		_x = {'x':np.dstack(ndict.ordered(x).values()).transpose((2,0,1))}
		_z = {'eps': np.dstack(ndict.ordered(z).values()).transpose((2,0,1))}
		return _x, _z
	
	def gwgz_to_numpy(self, gw, gz):
		_gz = {'eps'+str(i):gz['eps'][i] for i in range(self.n_steps)}
		return gw, _gz

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
	
	
	
	
	
	
	