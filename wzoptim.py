import numpy as np
import scipy
import scipy.optimize
import anglepy.ndict as ndict
import anglepy.BNModel as BNModel
import hmc
import time

# Training loop for MCEM
def loop_mcem(dostep, w, model, hook, hook_wavelength=2, n_iters=9999999):
	t_prev = time.time()
	
	z = [[]]
	logpxz = [[]]
	
	def getLoglik():
		if len(z[0]) < 4: return np.zeros(((0))), np.zeros(((0)))
		
		_z, _logpxz = hmc.mcmc_combine_samples(z[0], logpxz[0])
		ll, ll_var = hmc.compute_mcmc_likelihood(_z, _logpxz, len(z[0]))
		z[0] = []
		logpxz[0] = []
		return ll, ll_var
	
	for t in xrange(1, n_iters):
		_z, _logpxz = dostep(w)
		z[0].append(_z)
		logpxz[0].append(_logpxz)
		if t == 1 or time.time() - t_prev > hook_wavelength:
			ll, ll_var = getLoglik()
			hook(t, w, _z, ll, ll_var)
			t_prev = time.time()
	
	ll, ll_var = getLoglik()
	hook(n_iters-1, w, _z, ll, ll_var)
	
	print 'Optimization loop finished'

def lbfgs_wz(model, w, z, x, hook=None, maxiter=None):
	
	def f(y):
		_w, _z = ndict.unflatten_multiple(y, [w, z])
		logpx, logpz = model.logpxz(_w, _z, x)
		return - (logpx.sum() + logpz.sum())
	
	def fprime(y):
		_w, _z = ndict.unflatten_multiple(y, [w, z])
		logpx, logpz, gw, gz = model.dlogpxz_dwz(_w, _z, x)
		gwz = ndict.flatten_multiple((gw, gz))
		return - gwz
	
	t = [0, 0, time.time()]
	def callback(wz):
		if hook is None: return
		_w, _z = ndict.unflatten_multiple(wz, (w, z))
		t[1] += 1
		hook(t[1], _w, _z)
	
	x0 = ndict.flatten_multiple((w, z))
	xn, f, d = scipy.optimize.fmin_l_bfgs_b(func=f, x0=x0, fprime=fprime, m=100, iprint=0, callback=callback, maxiter=maxiter)
	
	#scipy.optimize.fmin_cg(f=f, x0=x0, fprime=fprime, full_output=True, callback=hook)
	#scipy.optimize.fmin_ncg(f=f, x0=x0, fprime=fprime, full_output=True, callback=hook)
	w, z = ndict.unflatten_multiple(xn, (w, z))
	if d['warnflag'] is 2:
		print 'warnflag:', d['warnflag']
		print d['task']
	return w, z
	

def step_batch_mcem(model_gen, x, z_mcmc, dostep_m, hmc_stepsize=1e-2, hmc_steps=20, m_steps=5):
	print 'Batch MCEM', hmc_stepsize, hmc_steps, m_steps
	
	n_batch = x.itervalues().next().shape[1]
	
	hmc_dostep = hmc.hmc_step_autotune(n_steps=hmc_steps, init_stepsize=hmc_stepsize)
	
	def doStep(w):
		
		def fgrad(_z):
			logpx, logpz, gw, gz = model_gen.dlogpxz_dwz(w, _z, x)
			return logpx + logpz, gz
		
		# E-step
		logpxz, acceptRate, stepsize = hmc_dostep(fgrad, z_mcmc)

		# M-step
		for _ in range(m_steps):
			dostep_m(w, z_mcmc)
		
		return z_mcmc.copy(), logpxz.copy() 
		
	return doStep

# HMC with both weights 'w' and latents 'z'
# Problem: stepsize of 'w' becomes really small
def step_hmc_wz(model, x, z, hmc_stepsize=1e-2, hmc_steps=20):
	print 'step_hmc_wz', hmc_stepsize, hmc_steps

	n_batch = x.itervalues().next().shape[1]
	
	hmc_dostep_z = hmc.hmc_step_autotune(n_steps=hmc_steps, init_stepsize=hmc_stepsize)
	hmc_dostep_w = hmc.hmc_step_autotune(n_steps=hmc_steps, init_stepsize=hmc_stepsize)
	
	def dostep(w):
		
		def fgrad_z(_z):
			logpx, logpz, gw, gz = model.dlogpxz_dwz(w, _z, x)
			return logpx + logpz, gz
		
		logpxz, acceptRate, stepsize = hmc_dostep_z(fgrad_z, z)

		shapes_w = ndict.getShapes(w)
		
		def vectorize(d):
			v = {}
			for i in d: v[i] = d[i].reshape((d[i].size, -1))
			return v
		
		def fgrad_w(_w):
			_w = ndict.setShapes(_w, shapes_w)
			logpx, logpz, gw, gz = model.dlogpxz_dwz(_w, z, x)
			gw = vectorize(gw)
			return logpx + logpz, gw
		
		_w = vectorize(w)
		hmc_dostep_w(fgrad_w, _w)
		
		return z.copy(), logpxz.copy() 
		
	return dostep


# Training loop for PVEM
def loop_pvem(dostep, w, model, hook, hook_wavelength=2, n_iters=9999999):
	
	t_prev = time.time()
	logpxz = n = 0
	
	for t in xrange(1, n_iters):
		z, _logpxz = dostep(w)
		logpxz += _logpxz
		n += 1
		if t == 1 or t == n_iters-1 or time.time() - t_prev > hook_wavelength:
			hook(t, w, z, np.average(logpxz/n))
			logpxz = n = 0
			t_prev = time.time()
	
	print 'Optimization loop finished'

# PVEM B (Predictive Variational EM)
def step_pvem(model_pred, model_gen, x, w_pred, n_batch=100, ada_stepsize=1e-1, warmup=10, reg=1e-8, convertImgs=False):
	print 'Predictive VEM', ada_stepsize
	
	hmc_steps=0
	hmc_dostep = hmc.hmc_step_autotune(n_steps=hmc_steps, init_stepsize=1e-1)
	
	# We're using adagrad stepsizes
	gw_pred_ss = ndict.cloneZeros(w_pred)
	gw_gen_ss = ndict.cloneZeros(model_gen.init_w())
	
	nsteps = [0]
	
	do_adagrad = True
	
	def doStep(w_gen):
		
		#def fgrad(_z):
		#	logpx, logpz, gw, gz = model_gen.dlogpxz_dwz(w, _z, x)
		#	return logpx + logpz, gz
		n_tot = x.itervalues().next().shape[1]
		idx_minibatch = np.random.randint(0, n_tot, n_batch)
		x_minibatch = {i:x[i][:,idx_minibatch] for i in x}
		if convertImgs: x_minibatch = {i:x_minibatch[i]/256. for i in x_minibatch}
			
		# step 1A: sample z ~ p(z|x) from model_pred
		_, z, _  = model_pred.gen_xz(w_pred, x_minibatch, {}, n_batch)
		
		# step 1B: update z using HMC
		def fgrad(_z):
			logpx, logpz, gw, gz = model_gen.dlogpxz_dwz(w_gen, _z, x_minibatch)
			return logpx + logpz, gz
		if (hmc_steps > 0):
			logpxz, _, _ = hmc_dostep(fgrad, z)

		def optimize(w, gw, gw_ss, stepsize):
			if do_adagrad:
				for i in gw:
					gw_ss[i] += gw[i]**2
					if nsteps[0] > warmup:
						w[i] += stepsize / np.sqrt(gw_ss[i]+reg) * gw[i]
			else:
				for i in gw:
					w[i] += 1e-4 * gw[i]
		
		# step 2: use z to update model_gen
		logpx_gen, logpz_gen, gw_gen, gz_gen = model_gen.dlogpxz_dwz(w_gen, z, x_minibatch)
		_, gw_prior = model_gen.dlogpw_dw(w_gen)
		gw = {i: gw_gen[i] + float(n_batch)/n_tot * gw_prior[i] for i in gw_gen}
		optimize(w_gen, gw, gw_gen_ss, ada_stepsize)
		
		# step 3: use gradients of model_gen to update model_pred
		_, logpz_pred, fd, gw_pred = model_pred.dfd_dw(w_pred, x_minibatch, z, gz_gen)
		_, gw_prior = model_pred.dlogpw_dw(w_pred)
		gw = {i: -gw_pred[i] + float(n_batch)/n_tot * gw_prior[i] for i in gw_pred}
		optimize(w_pred, gw, gw_pred_ss, ada_stepsize)
		
		nsteps[0] += 1
		
		return z.copy(), logpx_gen + logpz_gen - logpz_pred
		
	return doStep

