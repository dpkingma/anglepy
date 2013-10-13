import anglepy.hmc as hmc
import anglepy.ndict as ndict
import anglepy.BNModel as BNModel
import numpy as np
import scipy.optimize
import PIL.Image
import paramgraphics
import math, time
import scipy.stats

log = {
	'loglik': []
}

# Training loop
def loop(dostep, w, hook, hook_wavelength=2, n_iters=9999999):
	t_prev = time.time()
	
	logliks = [[]]
	
	def getLoglik():
		ll = np.asarray(logliks[0][-1]).mean()
		logliks[0] = []
		return ll
	for t in xrange(1, n_iters):
		loglik = dostep(w)
		logliks[0].append(loglik)
		if time.time() - t_prev > hook_wavelength:
			hook(t, w, getLoglik())
			t_prev = time.time()
	hook(n_iters-1, w, getLoglik())
	print 'Optimization loop finished'

def step_sgd(func, w, stepsize=1e-3):
	print 'SGD', stepsize
	batchi = [0]
	def doStep(w, z=None):
		v, gw = func.subgrad(batchi[0]%func.cardinality, w, z)
		
		for i in gw:
			w[i] += stepsize / func.blocksize * gw[i]
		batchi[0] += 1
		return v
	return doStep

# from: "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
# John Duchi et al (2010)
def step_adagrad(func, w, stepsize=1e-1):
	print 'Adagrad', stepsize
	
	# sum of squares of gradients and delta's of z's and w's
	gw_ss = ndict.cloneZeros(w)
	
	batchi = [0]
	
	def doStep(w, z=None):
		if z == None:
			v, gw = func.subgrad(batchi[0]%func.cardinality, w)
		else:
			v, gw = func.subgrad(batchi[0]%func.cardinality, w, z)
		for i in gw:
			#print i, np.sqrt(gw_ss[i]).max(), np.sqrt(gw_ss[i]).min()
			gw_ss[i] += gw[i]**2
			w[i] += stepsize / np.sqrt(gw_ss[i]) * gw[i]
		batchi[0] += 1
		
		return v
	
	return doStep

def step_adadelta(func, w, gamma=0.05, eps=1e-6):
	print 'Adadelta', gamma, eps
	
	# mean square of gradients and delta's of z's and w's
	gw_ms = ndict.cloneZeros(w)
	dw_ms = ndict.cloneZeros(w)
	dw = ndict.cloneZeros(w)
	
	batchi = [0]
	
	def doStep(w, z=None):
		if z == None:
			v, gw = func.subgrad(batchi[0]%func.cardinality, w)
		else:
			v, gw = func.subgrad(batchi[0]%func.cardinality, w, z)
		
		for i in gw:
			gw_ms[i] += gamma*(gw[i]**2 - gw_ms[i])
			dw[i] = np.sqrt(dw_ms[i] + eps)/np.sqrt(gw_ms[i] + eps) * gw[i]
			w[i] += dw[i]
			dw_ms[i] += gamma*(dw[i]**2 - dw_ms[i])
		batchi[0] += 1
		return v
	return doStep

def lbfgs(posterior, w, hook=None, hook_wavelength=2, m=100, maxiter=15000):
	print 'L-BFGS', m, maxiter
	
	cache = [1234,0,0]
	
	def eval(y):
		# TODO: solution for random seed in ipcluster
		# maybe just execute a remote np.random.seed(0) there?
		np.random.seed(0)
		_w = ndict.unflatten(y, w)
		logLik, gw = cache[1], cache[2]
		if np.linalg.norm(y) != cache[0]:
			logLik, gw = posterior.grad(_w)
			print logLik, np.linalg.norm(y)
			cache[0], cache[1], cache[2] = [np.linalg.norm(y), logLik, gw]
		return logLik, gw
	
	def f(y):
		logLik, gw = eval(y)
		return - logLik.mean()
	
	def fprime(y):
		logLik, gw = eval(y)
		#print '==================='
		#print '=>', joint, np.min(model.logpxz(_w, _z, x)), np.min(model.mclik(_w, x))
		#print '==================='
		gw = ndict.flatten(gw)
		log['loglik'].append(logLik)
		return - gw
	
	t = [0, 0, time.time()]
	def callback(wz):
		if hook is None: return
		t[1] += 1
		if time.time() - t[2] > hook_wavelength:
			_w = ndict.unflatten(wz, w)
			hook(t[1], _w)
			t[2] = time.time()
		#t[0] += 1
		#if t[0]%5 is not 0: return
		#if time.time() - t[2] < 1: return
		#t[2] = time.time()
	
	x0 = ndict.flatten(w)
	xn, f, d = scipy.optimize.fmin_l_bfgs_b(func=f, x0=x0, fprime=fprime, m=m, iprint=0, callback=callback, maxiter=maxiter)
	
	#scipy.optimize.fmin_cg(f=f, x0=x0, fprime=fprime, full_output=True, callback=hook)
	#scipy.optimize.fmin_ncg(f=f, x0=x0, fprime=fprime, full_output=True, callback=hook)
	w = ndict.unflatten(xn, w)
	print 'd: ', d
	if d['warnflag'] is 2:
		print 'warnflag:', d['warnflag']
		print d['task']
	return w
	
	