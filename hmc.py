import numpy as np
import matplotlib.pyplot as plt
import bnmodels.ndict as ndict
import scipy.stats
import time

log = {'acceptRate': [1],}

# Hybrid Monte Carlo sampler
# HMV move where x is a batch (rows=variables, columns=samples)
def hmc_step(fgrad, x0, _stepsize=1e-2, n_steps=20):
	
	# === INITIALIZE
	n_batch = x0.itervalues().next().shape[1]
	stepsize = (np.random.uniform(size=(1, n_batch)) < 0.5).astype(float)*2-1
	stepsize *= _stepsize
		
	if np.random.uniform() < 0.5:
		stepsize *= -1
	
	# Sample velocity
	vnew = {}
	for i in x0:
		vnew[i] = np.random.normal(size=x0[i].shape)
	
	# copy initial state 
	xnew = ndict.clone(x0)
	v0 = ndict.clone(vnew)
	
	# === LEAPFROG STEPS
	
	# Compute velocity at time (t + stepsize/2)
	# Compute position at time (t + stepsize)
	logpxz0, g = fgrad(xnew)
	for i in xnew:
		vnew[i] += 0.5 * stepsize * g[i]
		xnew[i] += stepsize * vnew[i]
	
	# Perform leapfrog steps
	for step in xrange(n_steps):
		logpxz, g = fgrad(xnew)
		for i in xnew:
			vnew[i] += stepsize * g[i]
			xnew[i] += stepsize * vnew[i]
	
	# Perform final half-step for velocity
	logpxz1, g = fgrad(xnew)
	for i in xnew:
		vnew[i] += 0.5 * stepsize * g[i]
	
	# === METROPOLIS-HASTINGS ACCEPT/REJECT
	
	# Compute old and new Hamiltonians

	k0 = 0
	k1 = 0
	for i in vnew:
		k0 += (v0[i]**2).sum(axis=0, keepdims=True)
		k1 += (vnew[i]**2).sum(axis=0, keepdims=True)
		
	h0 = -logpxz0 + 0.5 * k0
	h1 = -logpxz1 + 0.5 * k1
	
	#print logpxz0, k0, logpxz1, k1
	#print h0-h1
	
	# Perform Metropolis-Hasting step
	accept = np.exp(h0 - h1) >= np.random.uniform(size=h1.shape)
	accept = accept.astype(float)
	
	for i in xnew:
		accept2 = np.dot(np.ones((xnew[i].shape[0], 1)), accept)
		x0[i] = (1-accept2)*x0[i] + accept2 * xnew[i]
	
	# result: updated 'x0'
	logpxz = (1-accept)*logpxz0 + accept * logpxz1
	
	return logpxz, accept

def hmc_step_autotune(n_steps=20, init_stepsize=1e-2, target=0.9):
	alpha = 1.02
	stepsize = [init_stepsize]
	steps = [0]
	def dostep(fgrad, x):
		logpxz, accept = hmc_step(fgrad, x, stepsize[0], n_steps)
		acceptrate = accept.mean()
		if acceptrate < target:
			stepsize[0] /= alpha**float(accept.size)
		else:
			stepsize[0] *= alpha**float(accept.size)
		steps[0] += 1
		#print steps[0], 'stepsize:', acceptrate, stepsize[0], accept.size
		return logpxz, acceptrate, stepsize[0]
	return dostep

# Merge lists of z's and corresponding logpxz's into single matrices
def mcmc_combine_samples(z_list, logpxz_list, n_batch):
	n_list = len(logpxz_list)
	# Stack logpxz values into one matrix
	for i in range(n_list):
		logpxz_list[i] = logpxz_list[i].reshape((n_batch, 1))
	logpxz = np.hstack(logpxz_list).reshape((1, n_batch*n_list))
	
	# Stack x samples into one ndict
	z = {}
	for i in z_list[0].keys():
		list = []
		for _z in z_list:
			list.append(_z[i].reshape((-1, 1)))
		z[i] = np.hstack(list).reshape((-1, n_batch*n_list))
	
	return z, logpxz

# Compute the MCMC likelihood 
# z = MCMC samples 'z'
# logpxz = logp(x,z) corresponding to 'z'
# Unbiased: sample set of split for estimation of 'q' and 'p'
def compute_mcmc_likelihood(z, logpxz, n_batch, n_samples):
	
	if n_samples < 4: raise Exception("n_samples < 4")
	
	C = np.ones((n_samples,1))
	
	n_approx = int(n_samples/2)
	n_est = n_samples - n_approx
	
	def compute_logq_var(_z):
		_mean = _z[:,:n_approx].mean(axis=1, keepdims=True)
		_std = _z[:,:n_approx].std(axis=1, keepdims=True)
		
		if True:
			__std = _std[_std!=0]
			if len(__std) == 0:
				_stdmin = 1000
			else:
				_stdmin = np.min(__std)
				if _stdmin == 0: _stdmin = 1000
			_std = _std + (_std==0) * _stdmin
		_logpdf = scipy.stats.norm.logpdf(_z[:,n_approx:], _mean, _std)
		return _logpdf.reshape((-1, n_batch*n_est)).sum(axis=0, keepdims=True)

	def compute_logq_cov(_z):
		n_z = _z.shape[0]/n_batch
		logpdf = []
		for batch in range(n_batch):
			rows = range(n_z*batch,n_z*(batch+1))
			_mean = _z[rows,:n_approx].mean(axis=1, keepdims=True)
			_cov = np.cov(_z[rows,:n_approx])
			_logpdf = logpdf_mult_normal(_z[rows,n_approx:], _mean, _cov)
			logpdf.append(_logpdf.T)
		logpdf = np.vstack(logpdf)
		return logpdf.reshape((-1, n_batch*n_est))

	# PDF estimate with KDE (kernel density estimator)
	def compute_logq_kde(_z):
		n_z = _z.shape[0]/n_batch
		logpdf = []
		for batch in range(n_batch):
			rows = range(n_z*batch,n_z*(batch+1))
			#print(_z[rows,:n_approx])
			kde = scipy.stats.gaussian_kde(_z[rows,:n_approx])
			_logpdf = np.log(kde.evaluate(_z[rows,n_approx:]).reshape((1,-1)))
			logpdf.append(_logpdf.T)
		logpdf = np.vstack(logpdf)
		return logpdf.reshape((-1, n_batch*n_est))
	
	logq = 0
	for i in z:
		_z = z[i].reshape((-1, n_samples))
		if z[i].shape[0] == 1 or n_approx/z[i].shape[0] < 4:
			_logq = compute_logq_var(_z)
		else:
			if True:
				_logq = compute_logq_kde(_z)
			else:
				_logq = compute_logq_cov(_z)
		logq += _logq
	
	logpi = (logq - logpxz[:,n_approx*n_batch:]).reshape((n_batch, n_est))
	
	# This prevent some numerical issues
	logpi_max = logpi.max(axis=1, keepdims=True) #*0
	
	logpi_var = logpi.var(axis=1, keepdims=True)
	
	logpx = - (np.log(np.exp(logpi-logpi_max).mean(axis=1, keepdims=True)) + logpi_max).T
	
	# Computation of standard deviation
	# Using normal assumption of logpi distribution
	# logpi ~ Normal
	# Var[logpi_mean] ~= Var[logpx]/n_est
	# Var[logpi_var] ~= 2 * Var[logpx]**2/(n_est-1)
	# exp(logpi) ~ Log-normal
	# logpx = Mean[exp(logpi)] ~= -(logpi_mean + 0.5 * logpi_var)
	# So:
	# Var[logpx] = Var[logpi_mean] + 0.5 * Var[logpi_var]
	#            = Var[logpx]/n_est + Var[logpx]**2/(n_est-1)
	logpx_var = (logpi_var/n_est + logpi_var**2 / (n_est-1)).T
	
	#logpx_var = (2 * logpi_var/n_est).T
	
	if np.isnan(logpx).sum() > 0:
		raise Exception("NaN detected")
	
	return logpx, logpx_var

def logpdf_mult_normal(x, mean, cov):
	k = x.shape[0]
	if mean.shape[0] != k: raise Exception("x.shape[0] != mean.shape[0]")
	logpdf = - 0.5 * k * np.log(2*np.pi)
	logpdf -= 0.5 * np.log(np.linalg.det(cov))
	_x = x-mean
	logpdf += -0.5 * (np.dot(_x.T, np.linalg.inv(cov)) * _x.T).sum(axis=1, keepdims=True)
	
	if np.any(np.isinf(logpdf)):
		raise Exception('logpdf has infinities')
	
	return logpdf

# Alternative computation
# by assuming a log-normal distribution
# (sometimes gives better result)
def compute_mcmc_likelihood_lognormal(z, logpxz, n_batch, n_samples):
	
	C = np.ones((n_samples,1))
	
	logq = 0
	for i in z:
		_z = z[i].reshape((-1, n_samples))
		_mean = _z.mean(axis=1, keepdims=True)
		_std = _z.std(axis=1, keepdims=True)
		_logq = scipy.stats.norm.logpdf(_z, _mean, _std)
		logq += _logq.reshape((-1, n_batch*n_samples)).sum(axis=0, keepdims=True)
	
	logpi = (logq - logpxz).reshape((n_batch, n_samples))
	logpi_mean = logpi.mean(axis=1, keepdims=True)
	logpi_var = logpi.var(axis=1, keepdims=True)
	
	logpx = -(logpi_mean + 0.5 * logpi_var).T
	
	#inv_px = (1./n_samples * np.dot(pi, C)).T
	#inv_px = np.mean(pi, axis=1).T
	#inv_px_std = np.std(pi, axis=1).T
	
	return logpx

# Returns a log-likelihood estimator
def ll_estimator(model, _w, x, n_burnin=5, n_leaps=100, n_steps=10, stepsize=1e-3):
	#print 'MCMC Likelihood', stepsize, n_steps, n_leaps
	
	_, z_mcmc, _ = model.gen_xz(_w, x, {})
	
	n_batch_data = x.itervalues().next().shape[1]
	if model.n_batch != n_batch_data:
		raise Exception("{} != {}".format((model.n_batch, n_batch_data)))
	
	hmc_dostep = hmc_step_autotune(n_steps=n_steps, init_stepsize=stepsize)
	
	# Do one leapfrog step
	def doLeap(w):
		def fgrad(_z):
			logpx, logpz, gw, gz = model.dlogpxz_dwz(w, _z, x)
			return logpx + logpz, gz
		logpxz, _, _ = hmc_dostep(fgrad, z_mcmc)
		return logpxz
	
	# Estimate log-likelihood from samples
	def est_ll(w):
		for _ in range(n_burnin):
			doLeap(w)
		z_list = []
		logpxz_list = []
		for _ in range(n_leaps):
			logpxz = doLeap(w)
			z_list.append(z_mcmc.copy())
			logpxz_list.append(logpxz.copy())
			
		_z, _logpxz = mcmc_combine_samples(z_list, logpxz_list, model.n_batch)
		ll, var = compute_mcmc_likelihood(_z, _logpxz, model.n_batch, len(z_list))
		
		return ll, var
		
	return est_ll


'''
IDEA: Also implement this likelihood estimator:
Hamiltonian Annealed Importance Sampling
http://arxiv.org/pdf/1205.1925v1.pdf
'''

# Test the variance estimate
def test_variance_estimate():
	import bnmodels.models.DBN as DBN
	import math
	n_steps_dbn=1
	n_dims=64
	n_leaps=500
	
	model = DBN.create(n_z=n_dims, n_x=n_dims, n_steps=n_steps_dbn, n_batch=1, prior_sd=1)
	w_true = model.init_w(1)
	
	xkeys = ['x'+str(i) for i in range(n_steps_dbn)]
	x, _, _ = model.gen_xz(w_true, {}, {})
	
	for i in range(100):
		w = model.init_w(1)
		# Likelihood estimators
		est_train_ll = ll_estimator(model, w, x, n_burnin=5, n_leaps=n_leaps, n_steps=10, stepsize=1e-2)
		lls = []
		vars = []
		for i in range(50):
			est_train_ll(w)
		for i in range(50):
			lls, vars = est_train_ll(w)
			#print i, ll, std
			lls.append(lls.item(0))
			vars.append(vars.item(0))
		print np.asarray(lls).std(), math.sqrt(np.asarray(vars).mean())
	return

def test_variance_estimate2():
	import bnmodels.models.DBN as DBN
	import math
	n_steps_dbn=1
	n_dims=64
	
	model = DBN.create(n_z=n_dims, n_x=n_dims, n_steps=n_steps_dbn, n_batch=1, prior_sd=1)
	w_true = model.init_w(1)
	
	x, _, _ = model.gen_xz(w_true, {}, {})
	
	w = model.init_w(1)
	for j in range(4,100):
		# Likelihood estimators
		est_train_ll = ll_estimator(model, w, x, n_burnin=5, n_leaps=2**j, n_steps=10, stepsize=1e-2)
		lls = []
		vars = []
		for i in range(10):
			ll, var = est_train_ll(w)
			#print 'burnin', i, ll, std
		for i in range(50):
			ll, var = est_train_ll(w)
			#print 'sampling', i, ll, std
			lls.append(ll.item(0))
			vars.append(var.item(0))
		print 'n_leaps:', 2**j, np.asarray(lls).mean(), np.asarray(lls).std(), math.sqrt(np.asarray(vars).mean())
	return

# Sample autocorrelation
def autocorr(x):
	x -= x.mean()
	ac = np.correlate(x, x, mode='same')[len(x)/2:]
	return ac / ac[0]

# Effective sample size
# From: http://www.bayesian-inference.com/softwaredoc/ESS
def ess(x):
	x = autocorr(x)
	sum = 0
	for i in range(1,len(x)):
		#print i, x[i]
		if x[i] < 0.05: break
		sum += x[i]
	return len(x)/(1+2*sum)

