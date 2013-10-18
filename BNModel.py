import numpy as np
import theano
import theano.tensor as T
import math
import theano.compile
import anglepy.ndict as ndict
import anglepy.logpdfs
import inspect

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
class BNModel(object):
	
	def __init__(self, theano_warning='raise', hessian=True):
		
		theanofunction = lazytheanofunc('warn', mode='FAST_RUN')
		theanofunction_silent = lazytheanofunc('ignore', mode='FAST_RUN')
		
		# Create theano expressions
		# TODO: change order to (w, x, z) everywhere
		w, x, z = ordereddicts(self.variables())
		self.var_w, self.var_x, self.var_z, = w, x, z
		
		# Helper variables
		A = T.dmatrix('A') #np.ones((1, self.n_batch))
		
		# Get gradient symbols
		allvars = w.values()  + x.values() + z.values() + [A] # note: '+' concatenates lists
		
		# TODO: Split Hessian code from the core code (it's too rarely used), e.g. just in experiment script.
		if False and hessian:
			# Hessian of logpxz wrt z
			raise Exception("Needs fix: assumes fixed n_batch, which is not true anymore")
			n_batch = 100
			
			# Have to convert to vector and back to matrix because of stupid Theano hessian requirement
			z_concat = T.concatenate([T.flatten(z[i]) for i in z])
			
			# Translate back, we need the correct dimensions
			_, _z, _ = self.gen_xz(self.init_w(), {}, {}, n_batch=n_batch)
			shape_z = {i:_z[i].shape for i in _z}
			z = {}
			pointer = 0
			for i in _z:
				length = np.prod(shape_z[i])
				shape = shape_z[i]
				z[i] = T.reshape(z_concat[pointer:pointer+length], shape)
				pointer += length
			z = ndict.ordered(z)
		
		if False:
			# Put test values
			# needs fully implemented gen_xz(), which is not always the case
			# Also, the FD has no test values
			theano.config.compute_test_value = 'raise'
			_w = self.init_w()
			for i in _w: w[i].tag.test_value = _w[i]
			_x, _z, _ = self.gen_xz(_w, {}, {}, 10)
			_x, _z = self.xz_to_theano(_x, _z)
			for i in _x: x[i].tag.test_value = _x[i]
			for i in _z: z[i].tag.test_value = _z[i]
		
		logpw, logpx, logpz, dists = self.factors(w, x, z, A)
		
		# Complete-data likelihood estimate
		logpxz = logpx.sum() + logpz.sum()
		self.var_logpxz = logpxz
		
		self.f_logpxz = theanofunction(allvars, [logpx, logpz])
		
		dlogpxz_dwz = T.grad(logpxz, w.values() + z.values())
		self.f_dlogpxz_dwz = theanofunction(allvars, [logpx, logpz] + dlogpxz_dwz)
		#self.f_dlogpxz_dw = theanofunction(allvars, [logpxz] + dlogpxz_dw)
		#self.f_dlogpxz_dz = theanofunction(allvars, [logpxz] + dlogpxz_dz)
		
		# prior
		dlogpw_dw = T.grad(logpw, w.values(), disconnected_inputs='ignore')
		self.f_logpw = theanofunction(w.values(), logpw)
		self.f_dlogpw_dw = theanofunction(w.values(), [logpw] + dlogpw_dw)
		
		# distributions
		self.f_dists = {}
		for name in dists:
			_vars, dist = dists[name]
			self.f_dists[name] = theanofunction_silent(_vars, dist)
		
		if False:
			raise Exception("Code block needs refactoring: n_batch no longer a field of the model")
			# MC-LIKELIHOOD
			logpx_max = logpx.max()
			logpxmc = T.log(T.exp(logpx - logpx_max).sum()) + logpx_max - math.log(n_batch)
			self.f_logpxmc = theanofunction(allvars, logpxmc)
			dlogpxmc_dw = T.grad(logpxmc, w.values(), disconnected_inputs=theano_warning)
			self.f_dlogpxmc_dw = theanofunction(allvars, [logpxmc] + dlogpxmc_dw)
		
		if True and len(z) > 0:
			# Fisher divergence (FD)
			gz = T.grad(logpxz, z.values())
			gz2 = [T.dmatrix() for _ in gz]
			fd = 0
			for i in range(len(gz)):
				fd += T.sum((gz[i]-gz2[i])**2)
			dfd_dw = T.grad(fd, w.values())
			self.f_dfd_dw = theanofunction(allvars + gz2, [logpx, logpz, fd] + dfd_dw)
			
		if False and hessian:
			# Hessian of logpxz wrt z (works best with n_batch=1)
			hessian_z = theano.gradient.hessian(logpxz, z_concat)
			self.f_hessian_z = theanofunction(allvars, hessian_z)
		
	# NOTE: IT IS ESSENTIAL THAT DICTIONARIES OF SYMBOLIC VARS AND RESPECTIVE NUMPY VALUES HAVE THE SAME KEYS
	# (OTHERWISE FUNCTION ARGUMENTS ARE IN INCORRECT ORDER)
	
	def variables(self): raise NotImplementedError()
	def factors(self): raise NotImplementedError()
	def gen_xz(self): raise NotImplementedError()
	def init_w(self): raise NotImplementedError()
	
	# Prediction
	def distribution(self, w, x, z, name):
		x, z = self.xz_to_theano(x, z)
		w, z, x = ordereddicts((w, z, x))
		A = self.get_A(x)
		allvars = w.values() + x.values() + z.values() + [A]
		return self.f_dists[name](*allvars)
	
	# Numpy <-> Theano var conversion
	def xz_to_theano(self, x, z): return x, z
	def gwgz_to_numpy(self, gw, gz): return gw, gz
	
	# A = np.ones((1, n_batch))
	def get_A(self, x): return np.ones((1, x.itervalues().next().shape[1]))
		
	# Likelihood: logp(x,z|w)
	def logpxz(self, w, z, x):
		_x, _z = self.xz_to_theano(x, z)
		A = self.get_A(_x)
		allvars = w.values() + _x.values() + _z.values() + [A]
		logpx, logpz = self.f_logpxz(*allvars)
		if np.isnan(logpx).any() or np.isnan(logpz).any():
			print 'v: ', logpx, logpz
			print 'Values:'
			ndict.p(w)
			ndict.p(z)
			raise Exception("dlogpxz_dwz(): NaN found in gradients")
		
		return logpx, logpz
	
	# Gradient of logp(x,z|w) w.r.t. parameters and latent variables
	def dlogpxz_dwz(self, w, z, x):
		x, z = self.xz_to_theano(x, z)
		w, z, x = ordereddicts((w, z, x))
		A = self.get_A(x)
		allvars = w.values() + x.values() + z.values() + [A]
		r = self.f_dlogpxz_dwz(*allvars)
		logpx, logpz, gw, gz = r[0], r[1], dict(zip(w.keys(), r[2:2+len(w)])), dict(zip(z.keys(), r[2+len(w):]))
		
		if ndict.hasNaN(gw) or ndict.hasNaN(gz):
			if True:
				print 'NaN detected in gradients'
				raise Exception()
				for i in gw: gw[i][np.isnan(gw[i])] = 0
				for i in gz: gz[i][np.isnan(gz[i])] = 0
			else:
				
				print 'logpx: ', logpx
				print 'logpz: ', logpz
				print 'Values:'
				ndict.p(w)
				ndict.p(z)
				print 'Gradients:'
				ndict.p(gw)
				ndict.p(gz)
				raise Exception("dlogpxz_dwz(): NaN found in gradients")
		
		gw, gz = self.gwgz_to_numpy(gw, gz)
		return logpx, logpz, gw, gz
	
	'''
	# Gradient of logp(x,z|w) w.r.t. parameters
	def dlogpxz_dw(self, w, z, x):
		w, z, x = ordereddicts((w, z, x))
		r = self.f_dlogpxz_dw(*(w.values() + z.values() + x.values()))
		return r[0], dict(zip(w.keys(), r[1:]))
	
	# Gradient of logp(x,z|w) w.r.t. latent variables
	def dlogpxz_dz(self, w, z, x):
		w, z, x = ordereddicts((w, z, x))
		r = self.f_dlogpxz_dz(*(w.values() + z.values() + x.values()))
		return r[0], dict(zip(z.keys(), r[1:]))
	'''
	
	# Hessian of logpxz wrt z (works best with n_batch=1)
	def hessian_z(self, w, z, x):
		x, z = self.xz_to_theano(x, z)
		A = self.get_A(x)
		return self.f_hessian_z(*orderedvals((w, x, z))+[A])

	# Prior: logp(w)
	def logpw(self, w):
		logpw = self.f_logpw(*orderedvals((w,)))
		return logpw
	
	# Gradient of the prior: logp(w)
	def dlogpw_dw(self, w):
		w = ndict.ordered(w)
		r = self.f_dlogpw_dw(*(w.values()))
		return r[0], dict(zip(w.keys(), r[1:]))
	
	# MC likelihood: logp(x|w)
	def logpxmc(self, w, x, n_batch):
		x = self.tiled_x(x, n_batch)
		x, z, _ = self.gen_xz(w, x, {}, n_batch=x.shape[1])
		x, z = self.xz_to_theano(x, z)
		A = self.get_A(x)
		return self.f_logpxmc(*orderedvals((w, x, z))+[A])
	
	# Gradient of MC likelihood logp(x|w) w.r.t. parameters
	def dlogpxmc_dw(self, w, x, n_batch):
		x = self.tiled_x(x, n_batch)
		x, z, _ = self.gen_xz(w, x, {}, n_batch=x.shape[1])
		x, z = self.xz_to_theano(x, z)
		A = self.get_A(x)
		r = self.f_dlogpxmc_dw(*orderedvals((w, x, z))+[A])
		return r[0], dict(zip(ndict.ordered(w).keys(), r[1:]))
	
	# Gradient w.r.t. the Fisher divergence
	def dfd_dw(self, w, x, z, gz2):
		x, z = self.xz_to_theano(x, z)
		w, z, x, gz2 = ordereddicts((w, z, x, gz2))
		A = self.get_A(x)
		r = self.f_dfd_dw(*(w.values() + x.values() + z.values() + [A] + gz2.values()))
		logpx, logpz, fd, gw = r[0], r[1], r[2], dict(zip(w.keys(), r[3:3+len(w)]))
		
		if ndict.hasNaN(gw):
			if True:
				print 'NaN detected in gradients'
				raise Exception()
				for i in gw: gw[i][np.isnan(gw[i])] = 0
			else:
				
				print 'fd: ', fd
				print 'Values:'
				ndict.p(w)
				ndict.p(z)
				print 'Gradients:'
				ndict.p(gw)
				raise Exception("dfd_dw(): NaN found in gradients")
		
		gw, _ = self.gwgz_to_numpy(gw, {})
		return logpx, logpz, fd, gw
	
	# Helper function that creates tiled version of datapoint 'x' (* n_batch)
	def tiled_x(self, x, n_batch):
		x_tiled = {}
		for i in x:
			if (x[i].shape[1] != 1):
				raise Exception("{} {} {} ".format(x[i].shape[0], x[i].shape[1], n_batch))
			x_tiled[i] = np.dot(x[i], np.ones((1, n_batch)))
		return x_tiled
	
# converts normal dicts to ordered dicts, ordered by keys
def ordereddicts(ds):
	return [ndict.ordered(d) for d in ds]
def orderedvals(ds):
	vals = []
	for d in ds:
		vals += ndict.ordered(d).values()
	return vals

# Monte Carlo FuncLikelihood
class FuncLikelihoodMC():
	
	def __init__(self, x, model, n_batch):
		self.x = x
		self.model = model
		self.n_batch = n_batch
		n_train = x.itervalues().next().shape[1]
		if n_train%(self.n_batch) != 0: raise BaseException()
		self.blocksize = self.n_batch
		self.cardinality = n_train/self.blocksize
	
	def subval(self, i, w):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		return self.model.logpxmc(w, _x)
		
	def subgrad(self, i, w):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		logpx, gw = self.model.dlogpxmc_dw(w, _x)
		return logpx, gw
		
	def val(self, w):
		logpx = [self.subval(i, w) for i in range(self.cardinality)]
		return np.hstack(logpx)
	
	def grad(self, w):
		logpxi, gwi = tuple(zip(*[self.subgrad(i, w) for i in range(self.cardinality)]))
		return np.hstack(logpxi), ndict.sum(gwi)

# FuncLikelihood
class FuncLikelihood():
	
	def __init__(self, x, model, n_batch):
		self.x = x
		self.model = model
		self.n_batch = n_batch
		n_train = x.itervalues().next().shape[1]
		if n_train%(self.n_batch) != 0:
			print n_train, self.n_batch
			raise BaseException()
		self.blocksize = self.n_batch
		self.cardinality = n_train/self.blocksize
	
	def subval(self, i, w, z):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		_z = ndict.getCols(z, i*self.n_batch, (i+1)*self.n_batch)
		return self.model.logpxz(w, _z, _x)
	
	def subgrad(self, i, w, z):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		_z = ndict.getCols(z, i*self.n_batch, (i+1)*self.n_batch)
		logpx, logpz, g, _ = self.model.dlogpxz_dwz(w, _z, _x)
		return logpx, logpz, g
	
	def val(self, w, z):
		if self.cardinality==1: return self.subval(0, w, z)
		logpx, logpz = tuple(zip(*[self.subval(i, w, z) for i in range(self.cardinality)]))
		return np.hstack(logpx), np.hstack(logpz)
	
	def grad(self, w, z):
		if self.cardinality==1: return self.subgrad(0, w, z)
		logpxi, logpzi, gwi, _ = tuple(zip(*[self.subgrad(i, w, z) for i in range(self.cardinality)]))
		return np.hstack(logpxi), np.hstack(logpzi), ndict.sum(gwi)
	
# Parallel version of likelihood

# Before using, start ipython cluster, e.g.:
# shell>ipcluster start -n 4
from IPython.parallel.util import interactive
import IPython.parallel
class FuncLikelihoodPar():
	def __init__(self, x, model, n_batch):
		raise Exception("TODO")
		
		self.x = x
		self.c = c = IPython.parallel.Client()
		self.model = model
		self.n_batch = n_batch
		self.clustersize = len(c)
		
		print 'ipcluster size = '+str(self.clustersize)
		n_train = x.itervalues().next().shape[1]
		if n_train%(self.n_batch*len(c)) != 0: raise BaseException()
		self.blocksize = self.n_batch*len(c)
		self.cardinality = n_train/self.blocksize
		
		# Get pointers to slaves
		c.block = False
		# Remove namespaces on slaves
		c[:].clear()
		# Execute stuff on slaves
		module, function, args = self.model.constr
		c[:].push({'args':args,'x':x}).wait()
		commands = [
				'import os; cwd = os.getcwd()',
				'import sys; sys.path.append(\'../shared\')',
				'import anglepy.ndict as ndict',
				'import '+module,
				'my_n_batch = '+str(n_batch),
				'my_model = '+module+'.'+function+'(**args)'
		]
		for cmd in commands: c[:].execute(cmd).get()
		# Import data on slaves
		for i in range(len(c)):
			_x = ndict.getCols(x, i*(n_train/len(c)), (i+1)*(n_train/len(c)))
			c[i].push({'my_x':_x})
		c[:].pull(['my_x']).get()
		
	def subval(self, i, w, z):
		raise Exception("TODO")
		
		# Replaced my_model.nbatch with my_n_batch, this is UNTESTED
		
		@interactive
		def ll(w, z, k):
			_x = ndict.getCols(my_x, k*my_n_batch, (k+1)*my_n_batch) #@UndefinedVariable
			if z == None:
				return my_model.logpxmc(w, _x), None #@UndefinedVariable
			else:
				return my_model.logpxz(w, z, _x) #@UndefinedVariable
		
		tasks = []
		for j in range(len(self.c)):
			_z = z
			if _z != None:
				_z = ndict.getCols(z, j*self.n_batch, (j+1)*self.n_batch)
			tasks.append(self.c.load_balanced_view().apply_async(ll, w, _z, i))
		
		res = [task.get() for task in tasks]
		
		raise Exception("TODO: implementation with uncoupled logpx and logpz")
		return sum(res)
	
	def subgrad(self, i, w, z):
		
		@interactive
		def dlogpxz_dwz(w, z, k):
			_x = ndict.getCols(my_x, k*my_n_batch, (k+1)*my_n_batch).copy() #@UndefinedVariable
			if z == None:
				logpx, gw = my_model.dlogpxmc_dw(w, _x) #@UndefinedVariable
				return logpx, None, gw, None
			else:
				return my_model.dlogpxz_dwz(w, z, _x) #@UndefinedVariable
		
		tasks = []
		for j in range(len(self.c)):
			_z = z
			if _z != None:
				_z = ndict.getCols(z, j*self.n_batch, (j+1)*self.n_batch)
			tasks.append(self.c.load_balanced_view().apply_async(dlogpxz_dwz, w, _z, i))
		
		res = [task.get() for task in tasks]
		
		v, gw, gz = res[0]
		for k in range(1,len(self.c)):
			vi, gwi, gzi = res[k]
			v += vi
			for j in gw: gw[j] += gwi[j]
			for j in gz: gz[j] += gzi[j]
		return v, gw, gz
	

	def grad(self, w, z=None):
		v, gw, gz = self.subgrad(0, w, z)
		for i in range(1, self.cardinality):
			vi, gwi, gzi = self.subgrad(i, w, z)
			v += vi
			for j in gw: gw[j] += gwi[j]
			for j in gz: gz[j] += gzi[j]
		return v, gw, gz
	
	def val(self, w, z=None):
		logpx, logpz = self.subval(0, w, z)
		for i in range(1, self.cardinality):
			_logpx, _logpz = self.subval(i, w, z)
			logpx += _logpx
			logpz += _logpz
		return logpx, logpz
	
	def grad(self, w, z=None):
		logpx, logpz, gw, gz = self.subgrad(0, w, z)
		for i in range(1, self.cardinality):
			logpxi, logpzi, gwi, gzi = self.subgrad(i, w, z)
			logpx += logpxi
			logpz += logpzi
			for j in gw: gw[j] += gwi[j]
			for j in gz: gz[j] += gzi[j]
		return logpx, logpz, gw, gz
	
	# Helper function
	def getColsZX(self, w, z, i):
		_x = ndict.getCols(self.x, i*self.n_batch, (i+1)*self.n_batch)
		if z != None:
			_z = ndict.getCols(z, i*self.n_batch, (i+1)*self.n_batch)
		return _z, _x

# FuncPosterior	
class FuncPosterior():
	def __init__(self, likelihood, model):
		self.ll = likelihood
		self.model = model
		self.cardinality = likelihood.cardinality
		self.blocksize = likelihood.blocksize
	
	def subval(self, i, w, z):
		prior = self.model.logpw(w)
		prior_weight = 1. / float(self.ll.cardinality)
		logpx, logpz = self.ll.subval(i, w, z)
		return logpx.sum() + logpz.sum() + prior_weight * prior
	
	def subgrad(self, i, w, z):
		logpx, logpz, gw = self.ll.subgrad(i, w, z)
		prior, gw_prior = self.model.dlogpw_dw(w)
		prior_weight = 1. / float(self.ll.cardinality)
		for j in gw: gw[j] += prior_weight * gw_prior[j]
		return logpx.sum() + logpz.sum() + prior_weight * prior, gw
	
	def val(self, w, z={}):
		logpx, logpz = self.ll.val(w, z)
		return logpx.sum() + logpz.sum() + self.model.logpw(w)
	
	def grad(self, w, z={}):
		logpx, logpz, gw = self.ll.grad(w, z)
		prior, gw_prior = self.model.dlogpw_dw(w)
		for i in gw: gw[i] += gw_prior[i]
		return logpx.sum() + logpz.sum() + prior, gw
	
# FuncPosterior	
class FuncPosteriorMC():
	def __init__(self, likelihood, model):
		self.ll = likelihood
		self.model = model
		self.cardinality = likelihood.cardinality
		self.blocksize = likelihood.blocksize
	
	def subval(self, i, w):
		prior = self.model.logpw(w)
		prior_weight = 1. / float(self.ll.cardinality)
		logpx = self.ll.subval(i, w)
		return logpx.sum() + prior_weight * prior
	
	def subgrad(self, i, w):
		logpx, gw = self.ll.subgrad(i, w)
		v_prior, gw_prior = self.model.dlogpw_dw(w)
		prior_weight = 1. / float(self.ll.cardinality)
		v = logpx.sum() + prior_weight * v_prior
		for j in gw: gw[j] += prior_weight * gw_prior[j]
		return v, gw
	
	def val(self, w):
		logpx = self.ll.val(w)
		v = logpx.sum() + self.model.logpw(w)
		return v
	
	def grad(self, w):
		logpx, gw = self.ll.grad(w)
		v_prior, gw_prior = self.model.dlogpw_dw(w)
		v = logpx.sum() + v_prior
		for i in gw: gw[i] += gw_prior[i]
		return v, gw
	