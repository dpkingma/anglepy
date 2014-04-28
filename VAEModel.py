import numpy as np
import theano
import theano.tensor as T
import math
import theano.compile
import anglepy.ndict as ndict
import anglepy.logpdfs
import inspect

# ====
# VARIATIONAL AUTO-ENCODER MODEL
# ====

# Lazy function compilation
# (it only gets compiled when it's actually called)
def lazytheanofunc(on_unused_input='warn', mode='FAST_RUN'):
	def theanofunction(*args, **kwargs):
		f = [None]
		if not kwargs.has_key('on_unused_input'):
			kwargs['on_unused_input'] = on_unused_input
		if not kwargs.has_key('mode'):
			kwargs['mode'] = mode
		def func(*args2, **kwargs2):
			if f[0] == None:
				f[0] = theano.function(*args, **kwargs)
			return f[0](*args2, **kwargs2)
		return func
	return theanofunction

# Model
class VAEModel(object):
	
	def __init__(self, theano_warning='raise'):
		
		theanofunction = lazytheanofunc('warn', mode='FAST_RUN')
		theanofunction_silent = lazytheanofunc('ignore', mode='FAST_RUN')
		
		# Create theano expressions
		w, x, z = ndict.ordereddicts(self.variables())
		self.var_w, self.var_x, self.var_z, = w, x, z
		
		# Helper variables
		A = T.dmatrix('A')
		
		# Get gradient symbols
		allvars = w.values()  + x.values() + z.values() + [A] # note: '+' concatenates lists
				
		logpw, logpx, logpz, logqz, dists = self.factors(w, x, z, A)
		
		# Log-likelihood lower bound
		L = logpx.sum() + logpz.sum() - logqz.sum()
		self.f_L = theanofunction(allvars, [logpx, logpz, logqz])
		
		dL_dw = T.grad(L, w.values() + z.values())
		self.f_dL_dw = theanofunction(allvars, [logpx, logpz, logqz] + dL_dw)
		
		# prior
		dlogpw_dw = T.grad(logpw, w.values(), disconnected_inputs='ignore')
		self.f_logpw = theanofunction(w.values(), logpw)
		self.f_dlogpw_dw = theanofunction(w.values(), [logpw] + dlogpw_dw)
		
		# distributions
		self.f_dists = {}
		for name in dists:
			_vars, dist = dists[name]
			self.f_dists[name] = theanofunction_silent(_vars, dist)
					
	# NOTE: IT IS ESSENTIAL THAT DICTIONARIES OF SYMBOLIC VARS AND RESPECTIVE NUMPY VALUES HAVE THE SAME KEYS
	# (OTHERWISE FUNCTION ARGUMENTS ARE IN INCORRECT ORDER)
	
	def variables(self): raise NotImplementedError()
	def factors(self): raise NotImplementedError()
	def gen_xz(self): raise NotImplementedError()
	def init_w(self): raise NotImplementedError()
	
	# Prediction
	def distribution(self, w, x, z, name):
		x, z = self.xz_to_theano(x, z)
		w, x, z = ndict.ordereddicts((w, x, z))
		A = self.get_A(x)
		allvars = w.values() + x.values() + z.values() + [A]
		return self.f_dists[name](*allvars)
	
	# Numpy <-> Theano var conversion
	def xz_to_theano(self, x, z): return x, z
	def gw_to_numpy(self, gw): return gw
	
	# A = np.ones((1, n_batch))
	def get_A(self, x): return np.ones((1, x.itervalues().next().shape[1]))
		
	# Likelihood: logp(x,z|w)
	def L(self, w, x, z):
		x, z = self.xz_to_theano(x, z)
		w, z, x = ndict.ordereddicts((w, z, x))
		A = self.get_A(x)
		allvars = w.values() + x.values() + z.values() + [A]
		logpx, logpz, logqz = self.f_L(*allvars)
		
		if np.isnan(logpx).any() or np.isnan(logpz).any() or np.isnan(logqz).any():
			print 'v: ', logpx, logpz, logqz
			print 'Values:'
			ndict.p(w)
			ndict.p(x)
			ndict.p(z)
			raise Exception("delbo_dwz(): NaN found in gradients")
		
		return logpx, logpz, logqz
	
	# Gradient of logp(x,z|w) and logq(z) w.r.t. parameters
	def dL_dw(self, w, x, z):
		
		x, z = self.xz_to_theano(x, z)
		w, z, x = ndict.ordereddicts((w, z, x))
		A = self.get_A(x)
		allvars = w.values() + x.values() + z.values() + [A]
		r = self.f_dL_dw(*allvars)
		logpx, logpz, logqz, gw = r[0], r[1], r[2], dict(zip(w.keys(), r[3:3+len(w)]))
		
		if ndict.hasNaN(gw):
				print 'logpx: ', logpx
				print 'logpz: ', logpz
				print 'logqz: ', logqz
				print 'Values:'
				ndict.p(w)
				print 'Gradients:'
				ndict.p(gw)
				raise Exception("dL_dw(): NaN found in gradients")
		
		gw = self.gw_to_numpy(gw)
		return logpx, logpz, logqz, gw
	
	# Prior: logp(w)
	def logpw(self, w):
		logpw = self.f_logpw(*ndict.orderedvals((w,)))
		return logpw
	
	# Gradient of the prior: logp(w)
	def dlogpw_dw(self, w):
		w = ndict.ordered(w)
		r = self.f_dlogpw_dw(*(w.values()))
		return r[0], dict(zip(w.keys(), r[1:]))
	
	# Helper function that creates tiled version of datapoint 'x' (* n_batch)
	def tiled_x(self, x, n_batch):
		x_tiled = {}
		for i in x:
			if (x[i].shape[1] != 1):
				raise Exception("{} {} {} ".format(x[i].shape[0], x[i].shape[1], n_batch))
			x_tiled[i] = np.dot(x[i], np.ones((1, n_batch)))
		return x_tiled
	