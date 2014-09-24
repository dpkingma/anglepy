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
        lls = np.asarray(logliks[0])
        ll = lls.mean()
        logliks[0] = []
        return ll
    for t in xrange(1, n_iters):
        loglik = dostep(w)
        #print time.time() - t_prev, loglik
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
        if z is None: z = {}
        v, gw = func.subgrad(batchi[0]%func.n_minibatches, w, z)
        for i in gw:
            w[i] += stepsize / func.blocksize * gw[i]
        batchi[0] += 1
        return v
    return doStep


# from: "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
# John Duchi et al (2010)
def step_adagrad(func, w, stepsize=0.1, warmup=10, anneal=True, decay=0):
    print 'Adagrad', stepsize
    # sum of squares of gradients and delta's of z's and w's
    gw_ss = ndict.cloneZeros(w)
    batchi = [0]
    def doStep(w, z=None):
        if z is None: z = {}
        logpwxz, gw = func.subgrad(batchi[0]%func.n_minibatches, w, z)
        c = 1
        if not anneal:
            c = 1./ (batchi[0]+1)    
        for i in gw:
            #print i, np.sqrt(gw_ss[i]).max(), np.sqrt(gw_ss[i]).min()
            gw_ss[i] = (1-decay)*gw_ss[i] + gw[i]**2
            if batchi[0] < warmup: continue
            w[i] += stepsize / np.sqrt(gw_ss[i] * c + 1e-8) * gw[i]
        batchi[0] += 1
        return logpwxz
    return doStep

# RMSPROP objective
def step_rmsprop(w, model, x, prior_sd=1, n_batch=100, stepsize=1e-2, lambd=1e-2, warmup=10):
    print 'RMSprop', stepsize
    # sum of squares of gradients and delta's of z's and w's
    gw_ss = ndict.cloneZeros(w)
    n_datapoints = x.itervalues().next().shape[1]
    
    batchi = [0]
    
    def doStep(w):
        
        # Pick random minibatch
        idx = np.random.randint(0, n_datapoints, size=(n_batch,))
        _x = ndict.getColsFromIndices(x, idx)
        
        # Evaluate likelihood and its gradient
        logpx, _, gw, _ = model.dlogpxz_dwz(w, _x, {})
        
        for i in w:
            gw[i] *= n_datapoints / n_batch
        
        # Evalute prior and its gradient
        logpw = 0
        for i in w:
            logpw -= (.5 * (w[i]**2) / (prior_sd**2)).sum()
            gw[i] -= w[i] / (prior_sd**2)
        
        for i in gw:
            #print i, np.sqrt(gw_ss[i]).max(), np.sqrt(gw_ss[i]).min()
            gw_ss[i] += lambd * (gw[i]**2 - gw_ss[i])
            if batchi[0] < warmup: continue
            w[i] += stepsize * gw[i] / np.sqrt(gw_ss[i] + 1e-8)
        
        batchi[0] += 1
        
        return logpx + logpw

    return doStep

# RMSPROP objective
def step_woga(w, funcs, func_holdout, stepsize=1e-2, lambd=1e-2, warmup=10):
    print 'WOGA', stepsize
    
    batchi = [0]
    
    m1 = [ndict.cloneZeros(w) for i in range(len(funcs))]
    m2 = [ndict.cloneZeros(w) for i in range(len(funcs))]
    m1_holdout = ndict.cloneZeros(w)
    m2_holdout = ndict.cloneZeros(w)
    
    def doStep(w):
        
        f_holdout, gw_holdout = func_holdout(w)
        gw_holdout_norm = 0
        gw_holdout_effective = ndict.clone(gw_holdout)
        for i in w:
            m1_holdout[i] += lambd * (gw_holdout[i] - m1_holdout[i])
            m2_holdout[i] += lambd * (gw_holdout[i]**2 - m2_holdout[i])
            gw_holdout_effective[i] /= np.sqrt(m2_holdout[i] + 1e-8)
            gw_holdout_norm += (gw_holdout_effective[i]**2).sum()
        gw_holdout_norm = np.sqrt(gw_holdout_norm)
        
        f_tot = 0
        gw_tot = ndict.cloneZeros(w)
        alphas = []
        for j in range(len(funcs)):
            f, gw = funcs[j](w)
            f_tot += f
            
            gw_norm = 0
            gw_effective = ndict.clone(gw)
            for i in w:
                # Update first and second moments
                m1[j][i] += lambd * (gw[i] - m1[j][i])
                m2[j][i] += lambd * (gw[i]**2 - m2[j][i])
                gw_effective[i] /= np.sqrt(m2[j][i] + 1e-8)
                gw_norm += (gw_effective[i]**2).sum()
            gw_norm = np.sqrt(gw_norm)
            
            # Compute dot product with holdout gradient
            alpha = 0
            for i in w:
                alpha += (gw_effective[i] * gw_holdout_effective[i]).sum()
            alpha /= gw_holdout_norm * gw_norm
            
            alphas.append(alpha)
            
            #alpha = (alpha > 0) * 1.0
            
            for i in w:
                # Accumulate gradient of subobjective
                gw_tot[i] += alpha * gw[i] / np.sqrt(m2[j][i] + 1e-8)
            
        #print 'alphas:', alphas
        
        if batchi[0] > warmup:
            for i in w:
                w[i] += stepsize * gw_tot[i]
        
        
        batchi[0] += 1
        
        return f_tot

    return doStep

# AdaDelta (Matt Zeiler)
def step_adadelta(func, w, gamma=0.05, eps=1e-6):
    print 'Adadelta', gamma, eps
    
    # mean square of gradients and delta's of z's
    gw_ms = ndict.cloneZeros(w)
    dw_ms = ndict.cloneZeros(w)
    dw = ndict.cloneZeros(w)
    
    batchi = [0]
    
    def doStep(w, z=None):
        if z == None: z = {}
        v, gw = func.subgrad(batchi[0]%func.n_minibatches, w, z)
        
        for i in gw:
            gw_ms[i] += gamma*(gw[i]**2 - gw_ms[i])
            dw[i] = np.sqrt(dw_ms[i] + eps)/np.sqrt(gw_ms[i] + eps) * gw[i]
            w[i] += dw[i]
            dw_ms[i] += gamma*(dw[i]**2 - dw_ms[i])
        batchi[0] += 1
        return v
    return doStep

# L-BFGS
def lbfgs(posterior, w, hook=None, hook_wavelength=2, m=100, maxiter=15000):
    print 'L-BFGS', m, maxiter
    
    cache = [1234,0,0]
    
    def eval(y):
        # TODO: solution for random seed in ipcluster
        # maybe just execute a remote np.random.seed(0) there?
        np.random.seed(0)
        _w = ndict.unflatten(y, w)
        logpw, gw = cache[1], cache[2]
        if np.linalg.norm(y) != cache[0]:
            logpw, gw = posterior.grad(_w)
            #print logpw, np.linalg.norm(y)
            cache[0], cache[1], cache[2] = [np.linalg.norm(y), logpw, gw]
        return logpw, gw
    
    def f(y):
        logLik, gw = eval(y)
        return - logLik.mean()
    
    def fprime(y):
        logLik, gw = eval(y)
        #print '==================='
        #print '=>', joint, np.min(model.logpxz(_w, x, _z)), np.min(model.mclik(_w, x))
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
            hook(t[1], _w, cache[1]) # num_its, w, logpw
            t[2] = time.time()
    
    x0 = ndict.flatten(w)
    xn, f, d = scipy.optimize.fmin_l_bfgs_b(func=f, x0=x0, fprime=fprime, m=m, iprint=0, callback=callback, maxiter=maxiter)
    
    #scipy.optimize.fmin_cg(f=f, x0=x0, fprime=fprime, full_output=True, callback=hook)
    #scipy.optimize.fmin_ncg(f=f, x0=x0, fprime=fprime, full_output=True, callback=hook)
    w = ndict.unflatten(xn, w)
    #print 'd: ', d
    if d['warnflag'] is 2:
        print 'warnflag:', d['warnflag']
        print d['task']
    info = d
    return w, info
        

# SGVB with Adagrad stepsizes
def step_adasgvb(w, logsd, x, model, var='diag', antithetic=False, init_logsd=0, prior_sd=1, n_batch=1, n_subbatch=100, stepsize=1e-2, warmup=10, momw=0.75, momsd=0.75, anneal=False, sgd=False):
    print "SGVB + Adagrad", var, antithetic, init_logsd, prior_sd, n_batch, n_subbatch, stepsize, warmup, momw, momsd, anneal, sgd
    
    # w and logsd are the variational mean and log-variance that are learned

    g_w_ss = ndict.cloneZeros(w)
    mom_w = ndict.cloneZeros(w)

    if var == 'diag' or var == 'row_isotropic':
        #logsd = ndict.cloneZeros(w)
        for i in w: logsd[i] += init_logsd
        g_logsd_ss = ndict.cloneZeros(w)
        mom_logsd = ndict.cloneZeros(w)
    elif var == 'isotropic':
        logsd = {i: init_logsd for i in w}
        g_logsd_ss = {i: 0 for i in w}
        mom_logsd = {i: 0 for i in w}
    else: raise Exception("Unknown variance type")
    
    n_datapoints = x.itervalues().next().shape[1]
    
    batchi = [0]
    def doStep(w, z=None):
        if z is not None: raise Exception()
        
        L = [0] # Lower bound
        g_mean = ndict.cloneZeros(w)
        if var == 'diag' or var == 'row_isotropic':
            g_logsd = ndict.cloneZeros(w)
        elif var == 'isotropic':
            g_logsd = {i:0 for i in w}
        
        # Loop over random datapoints
        for l in range(n_batch):
            
            # Pick random datapoint
            idx = np.random.randint(0, n_datapoints, size=(n_subbatch,))
            _x = ndict.getColsFromIndices(x, idx)
            
            # Function that adds gradients for given noise eps
            def add_grad(eps):
                # Compute noisy weights
                _w = {i: w[i] + np.exp(logsd[i]) * eps[i] for i in w}
                # Compute gradients of log p(x|theta) w.r.t. w
                logpx, logpz, g_w, g_z = model.dlogpxz_dwz(_w, _x, {})        
                for i in w:
                    g_mean[i] += g_w[i]
                    if var == 'diag' or var == 'row_isotropic':
                        g_logsd[i] += g_w[i] * eps[i] * np.exp(logsd[i])
                    elif var == 'isotropic':
                        g_logsd[i] += (g_w[i] * eps[i]).sum() * np.exp(logsd[i])
                    else: raise Exception()
                
                L[0] += logpx.sum() + logpz.sum()
            
            # Gradients with generated noise
            eps = {i: np.random.standard_normal(size=w[i].shape) for i in w}
            if sgd: eps = {i: np.zeros(w[i].shape) for i in w}
            add_grad(eps)
            
            # Gradient with negative of noise
            if antithetic:
                for i in eps: eps[i] *= -1
                add_grad(eps)
        
        L = L[0]        
        L *= float(n_datapoints) / float(n_subbatch) / float(n_batch)
        if antithetic: L /= 2
        
        for i in w:
            g_mean[i] *= float(n_datapoints) / (float(n_subbatch) * float(n_batch))
            g_logsd[i] *= float(n_datapoints) / (float(n_subbatch) * float(n_batch))
            if antithetic:
                g_mean[i] /= 2
                g_logsd[i] /= 2
            
            # Prior
            g_mean[i] += - w[i] / (prior_sd**2)
            g_logsd[i] += - np.exp(2 * logsd[i]) / (prior_sd**2)
            L += - (w[i]**2 + np.exp(2 * logsd[i])).sum() / (2 * prior_sd**2)
            L += - 0.5 * np.log(2 * np.pi * prior_sd**2) * float(w[i].size)
            
            # Entropy
            L += float(w[i].size) * 0.5 * math.log(2 * math.pi * np.pi)
            if var == 'diag' or var == 'row_isotropic':
                g_logsd[i] += 1 # dH(q)/d[logsd] = 1 (nice!)
                L += logsd[i].sum()
            elif var == 'isotropic':
                g_logsd[i] += float(w[i].size) # dH(q)/d[logsd] = 1 (nice!)
                L += logsd[i] * float(w[i].size)
            else: raise Exception()
            
        # Update variational parameters
        c = 1
        if not anneal:
            c = 1./ (batchi[0] + 1)
        
        # For isotropic row variance, sum gradients per row
        if var == 'row_isotropic':
            for i in w:
                g_sum = g_logsd[i].sum(axis=1).reshape(w[i].shape[0], 1)
                g_logsd[i] = np.dot(g_sum, np.ones((1, w[i].shape[1])))
        
        for i in w:
            #print i, np.sqrt(gw_ss[i]).max(), np.sqrt(gw_ss[i]).min()
            g_w_ss[i] += g_mean[i]**2
            g_logsd_ss[i] += g_logsd[i]**2
            
            mom_w[i] += (1-momw) * (g_mean[i] - mom_w[i])
            mom_logsd[i] += (1-momsd) * (g_logsd[i] - mom_logsd[i])
            
            if batchi[0] < warmup: continue
            
            w[i] += stepsize / np.sqrt(g_w_ss[i] * c + 1e-8) * mom_w[i]
            logsd[i] += stepsize / np.sqrt(g_logsd_ss[i] * c + 1e-8) * mom_logsd[i]            
            
        batchi[0] += 1
        
        return L

    return doStep


# SGVB with adaptive stepsizes
# and with epsilon as control variate
# NOTE: EXPERIMENTS DO NOT SEE FASTER CONVERGENCE WITH THESE CONTROL VARIATES
def step_adasgvb2(w, logsd, x, model, var='diag', negNoise=False, init_logsd=0, prior_sd=1, n_batch=1, n_subbatch=100, stepsize=1e-2, warmup=10, momw=0.75, momsd=0.75, anneal=False, sgd=False):
    print "SGVB + Adagrad", var, negNoise, init_logsd, prior_sd, n_batch, n_subbatch, stepsize, warmup, momw, momsd, anneal, sgd
    
    
    # w and logsd are the variational mean and log-variance that are learned
    
    g_w_ss = ndict.cloneZeros(w) # sum-of-squares for adagrad
    mom_w = ndict.cloneZeros(w) # momentum
    
    cv_lr = 0.1 # learning rate for control variates
    cov_mean = ndict.cloneZeros(w)
    var_mean = ndict.cloneZeros(w)
    cov_logsd = ndict.cloneZeros(w)
    var_logsd = ndict.cloneZeros(w)
    
    if var != 'diag':
        raise Exception('Didnt write control variate code for non-diag variance yet')
    
    if var == 'diag' or var == 'row_isotropic':
        #logsd = ndict.cloneZeros(w)
        for i in w: logsd[i] += init_logsd
        g_logsd_ss = ndict.cloneZeros(w)
        mom_logsd = ndict.cloneZeros(w)
    elif var == 'isotropic':
        logsd = {i: init_logsd for i in w}
        g_logsd_ss = {i: 0 for i in w}
        mom_logsd = {i: 0 for i in w}
    else: raise Exception("Unknown variance type")
    
    n_datapoints = x.itervalues().next().shape[1]
    
    batchi = [0]
    def doStep(w, z=None):
        if z is not None: raise Exception()
        
        L = [0] # Lower bound
        g_mean = ndict.cloneZeros(w)
        if var == 'diag' or var == 'row_isotropic':
            g_logsd = ndict.cloneZeros(w)
        elif var == 'isotropic':
            g_logsd = {i:0 for i in w}
        
        # Loop over random datapoints
        for l in range(n_batch):
            
            # Pick random datapoint
            idx = np.random.randint(0, n_datapoints, size=(n_subbatch,))
            _x = ndict.getColsFromIndices(x, idx)
            
            # Function that adds gradients for given noise eps
            def add_grad(eps):
                # Compute noisy weights
                _w = {i: w[i] + np.exp(logsd[i]) * eps[i] for i in w}
                # Compute gradients of log p(x|theta) w.r.t. w
                logpx, logpz, g_w, g_z = model.dlogpxz_dwz(_w, _x, {})        
                for i in w:
                    cv = (_w[i] - w[i]) / np.exp(2*logsd[i])  #control variate
                    cov_mean[i] += cv_lr * (g_w[i]*cv - cov_mean[i])
                    var_mean[i] += cv_lr * (cv**2 - var_mean[i])
                    g_mean[i] += g_w[i] - cov_mean[i]/var_mean[i] * cv
                    
                    if var == 'diag' or var == 'row_isotropic':
                        grad = g_w[i] * eps[i] * np.exp(logsd[i])
                        cv = cv - 1 # this control variate (c.v.) is really similar to the c.v. for the mean!
                        cov_logsd[i] += cv_lr * (grad*cv - cov_logsd[i])
                        var_logsd[i] += cv_lr * (cv**2  - var_logsd[i])
                        g_logsd[i] += grad - cov_logsd[i]/var_logsd[i] * cv
                    elif var == 'isotropic':
                        g_logsd[i] += (g_w[i] * eps[i]).sum() * np.exp(logsd[i])
                    else: raise Exception()
                    
                L[0] += logpx.sum() + logpz.sum()
            
            # Gradients with generated noise
            eps = {i: np.random.standard_normal(size=w[i].shape) for i in w}
            if sgd: eps = {i: np.zeros(w[i].shape) for i in w}
            add_grad(eps)
            
            # Gradient with negative of noise
            if negNoise:
                for i in eps: eps[i] *= -1
                add_grad(eps)
        
        L = L[0]        
        L *= float(n_datapoints) / float(n_subbatch) / float(n_batch)
        if negNoise: L /= 2
        
        for i in w:
            c = float(n_datapoints) / (float(n_subbatch) * float(n_batch))
            if negNoise: c /= 2
            g_mean[i] *= c
            g_logsd[i] *= c
                        
            # Prior
            g_mean[i] += - w[i] / (prior_sd**2)
            g_logsd[i] += - np.exp(2 * logsd[i]) / (prior_sd**2)
            L += - (w[i]**2 + np.exp(2 * logsd[i])).sum() / (2 * prior_sd**2)
            L += - 0.5 * np.log(2 * np.pi * prior_sd**2) * float(w[i].size)
            
            # Entropy
            L += float(w[i].size) * 0.5 * math.log(2 * math.pi * np.pi)
            if var == 'diag' or var == 'row_isotropic':
                g_logsd[i] += 1 # dH(q)/d[logsd] = 1 (nice!)
                L += logsd[i].sum()
            elif var == 'isotropic':
                g_logsd[i] += float(w[i].size) # dH(q)/d[logsd] = 1 (nice!)
                L += logsd[i] * float(w[i].size)
            else: raise Exception()
            
        # Update variational parameters
        c = 1
        if not anneal:
            c = 1./ (batchi[0] + 1)
        
        # For isotropic row variance, sum gradients per row
        if var == 'row_isotropic':
            for i in w:
                g_sum = g_logsd[i].sum(axis=1).reshape(w[i].shape[0], 1)
                g_logsd[i] = np.dot(g_sum, np.ones((1, w[i].shape[1])))
        
        for i in w:
            #print i, np.sqrt(gw_ss[i]).max(), np.sqrt(gw_ss[i]).min()
            g_w_ss[i] += g_mean[i]**2
            g_logsd_ss[i] += g_logsd[i]**2
            
            mom_w[i] += (1-momw) * (g_mean[i] - mom_w[i])
            mom_logsd[i] += (1-momsd) * (g_logsd[i] - mom_logsd[i])
            
            if batchi[0] < warmup: continue
            
            w[i] += stepsize / np.sqrt(g_w_ss[i] * c + 1e-8) * mom_w[i]
            logsd[i] += stepsize / np.sqrt(g_logsd_ss[i] * c + 1e-8) * mom_logsd[i]            
            
        batchi[0] += 1
        
        #print cov_mean['b0']/var_mean['b0']
        
        return L

    return doStep


# Fixed-Form VB (Salimans)
# With a fully factorized posterior distribution
def step_ffvb(w, logsd, x, model):
    
    # Initialize natural params
    eta1 = {} # natural param 1 = mu/sigma^2
    eta2 = {} # natural param 2= -1/(2*sigma^2)
    C11 = {}
    C12 = {}
    C21 = {}
    C22 = {}
    g1 = {}
    g2 = {}
    for i in w:
        eta1[i] = w[i]/np.exp(2*logsd[i])   #eta1 = mu/sigma^2
        eta2[i] = -1/(2*np.exp(2*logsd[i])) #eta2 = -1/(2*sigma^2)
        C11[i] = np.ones(w[i].shape)
        C12[i] = np.zeros(w[i].shape)
        C21[i] = np.zeros(w[i].shape)
        C22[i] = np.ones(w[i].shape)
        g1[i] = np.zeros(w[i].shape)#C11[i]*eta1[i]
        g2[i] = np.zeros(w[i].shape)#C22[i]*eta2[i]
    
    stepsize = 1e-3
    n_datapoints = x.itervalues().next().shape[1]
    n_minibatch = 100
    
    stochastic=True
    if not stochastic:
        n_minibatch = n_datapoints
    
    from scipy.stats import norm
    
    iter = [0]
    
    def doStep(w):
        
        LB = 0 #Lower bound
        
        idx = np.random.randint(0, n_datapoints, size=(n_minibatch,))
        _x = ndict.getColsFromIndices(x, idx)
        if not stochastic:
            _x = x
        
        # Draw sample _w from posterior q(w;eta1,eta2)
        eps = {}
        _w = {}
        for i in w:
            eps[i] = np.random.standard_normal(size=w[i].shape)
            _w[i] = w[i] + np.exp(logsd[i])*eps[i]
            LB += (0.5 + 0.5 * np.log(2 * np.pi) + logsd[i]).sum()
        
        # Compute L = log p(x,w)
        logpx, logpz, gw, gz = model.dlogpxz_dwz(_w, _x, {})
        logpw, gw2 = model.dlogpw_dw(_w)
        for i in gw: gw[i] = (float(n_datapoints) / float(n_minibatch)) * gw[i] + gw2[i]
        
        L = (logpx.sum() + logpz.sum()) * float(n_datapoints) / float(n_minibatch)
        L += logpw.sum()
        LB += L
        
        # Update params
        for i in w:
            
            # Noisy estimates g' and C'
            # l = log p(x,w)
            # w = mean + sigma * eps = - eta1/(2*eta2) - 1/(2*eta2) * eps = - (eta1+eps)/(2*eta2)
            # dw/deta1 = -1/(2*eta2)
            # dw/deta2 = (eta1 + eps)/(2*eta2^2)
            # g1hat = dl/deta1 = dl/dw dw/deta1 = gw[i] * dw/deta1
            # g2hat = dl/deta2 = dl/dw dw/deta2
            dwdeta1 = -1/(2*eta2[i])
            dwdeta2 = (eta1[i] + eps[i]) / (2*eta2[i]**2)
            g1hat = gw[i] * dwdeta1
            g2hat = gw[i] * dwdeta2
            
            # C11hat = dw/dw * dw/deta1
            # C12hat = d(w**2)/dw * dw/deta1
            # C21hat = dw/dw * dw/deta2
            # C22hat = d(w**2)/dw * dw/deta2
            C11hat = dwdeta1
            C12hat = 2 * _w[i] * dwdeta1
            C21hat = dwdeta2
            C22hat = 2 * _w[i] * dwdeta2
            
            if i == 'b0':
                #print g1['b0'][0].T, g1hat[0], g2['b0'][0].T, g2hat[0]
                #print C11['b0'][0].T, C11hat[0], C22['b0'][0].T, C22hat[0]
                #print T1[0], T2[0], logsd[i][0]
                #print iter[0], w[i][0], logsd[i][0], w[i][1], logsd[i][1], w0, L
                pass
            
            # Update running averages of g and C
            if True:
                g1[i] = (1-stepsize)*g1[i] + stepsize*g1hat
                g2[i] = (1-stepsize)*g2[i] + stepsize*g2hat
                
                C11[i] = (1-stepsize)*C11[i] + stepsize*C11hat
                C12[i] = (1-stepsize)*C12[i] + stepsize*C12hat
                C21[i] = (1-stepsize)*C21[i] + stepsize*C21hat
                C22[i] = (1-stepsize)*C22[i] + stepsize*C22hat
                
            else:
                g1[i] = (1-stepsize)*g1[i] + g1hat
                g2[i] = (1-stepsize)*g2[i] + g2hat
                
                C11[i] = (1-stepsize)*C11[i] + C11hat
                C12[i] = (1-stepsize)*C12[i] + C12hat
                C21[i] = (1-stepsize)*C21[i] + C21hat
                C22[i] = (1-stepsize)*C22[i] + C22hat
                
            if iter[0] > 0.1/stepsize:
                # Compute parameters given current g and C
                # eta = C^-1 g
                # => eta1 = det(C) * (C22[i] * g1[i] - C12[i] * g2[i])
                # => eta2 = det(C) * (-C21[i] * g1[i] + C11[i] * g2[i])
                det = 1/(C11[i] * C22[i] - C12[i] * C21[i])
                eta1[i] = det * (C22[i] * g1[i] - C12[i] * g2[i])
                eta2[i] = det * (-C21[i] * g1[i] + C11[i] * g2[i])
                
                eta2[i] = -np.abs(eta2[i])
                
                # Map natural parameters to mean and variance parameters
                w[i] = - eta1[i]/(2*eta2[i])
                logsd[i] = 0.5 * np.log( - 1/(2*eta2[i]))
                
                if np.isnan(w[i]).sum() > 0:
                    print 'w', i, np.isnan(w[i]).sum()
                    raise Exception()
                
                if np.isnan(logsd[i]).sum() > 0:
                    print 'logsd', i, np.isnan(logsd[i]).sum()
                    raise Exception()
                
                
        iter[0] += 1
        
        return LB
    
    return doStep


    