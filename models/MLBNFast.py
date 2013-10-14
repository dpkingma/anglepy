import numpy as np
import theano
import theano.tensor as T
import collections as C
import anglepy.BNModel as BNModel
import anglepy.logpdfs as logpdfs
import math, inspect

# Fast MLBN
def create(n_hidden, n_output, n_batch, prior_sd=0.1, noMiddleEps=False):
	raise Exception("Needs refactoring!")

	constr = (__name__, inspect.stack()[0][3], locals())
	
	def functions(w, z, x):
		
		# Define symbolic program
		A = np.ones((1, n_batch))
		B =  np.ones((n_batch,1))
		C = np.ones((1, n_output))
		hidden = []
		hidden.append(z['eps0'])
		
		if noMiddleEps: gate = 0;
		else: gate = 1
		
		for i in range(1, len(n_hidden)):
			hidden.append(T.tanh(T.dot(w['w%d'%i], hidden[i-1]) + T.dot(w['b'+str(i)], A)) + gate * z['eps'+str(i)] * T.dot(T.exp(w['logsd'+str(i)]), A))
		p = 0.5 + 0.5 * T.tanh(T.dot(w['wout'], hidden[-1]) + T.dot(w['bout'], A)) # p = p(X=1)
		
		if False:
			# NOTE: code below should be correct but gives obscure Theano error
			
			# duplicate columns of p
			xt = T.dot(x['x'].reshape((n_output*n_batch, 1)), A).reshape((n_output, n_batch*n_batch))
			
			# tile p
			p = p.copy().T.copy()
			p = p.reshape((1, n_output*n_batch)).copy()
			p = T.dot(B, p).reshape((n_batch*n_batch, n_output)).copy().T.copy()
			
			logpx = T.log(xt * p + (1-xt)*(1-p))
			logpx = T.dot(np.ones((1, n_output)), logpx)
			
			logpx = logpx.reshape((n_batch,n_batch))
			logpx_means = logpx.max(axis=1).dimshuffle(0, 'x')
			pxz = T.exp(logpx - T.dot(logpx_means, A))
			logpx = (T.log(T.dot(pxz, 1./n_batch * B)) + logpx_means).reshape((1,n_batch))
		
		if False:
			# NOTE: this alternative method seems to work, but is super slow to compile
			logpx = []
			for i in range(n_batch):
				xi = T.dot(x['x'][:,i:i+1], A)
				logpxi = T.log(xi * p + (1-xi)*(1-p))
				logpxi = T.dot(np.ones((1, n_output)), logpxi)
				logpxi_max = logpxi.max()#logpx.max(axis=1).dimshuffle(0, 'x')
				pxz = T.exp(logpxi - logpxi_max)
				logpxi = T.log(T.dot(pxz, 1./n_batch * B)) + logpxi_max
				#T.basic.set_subtensor(logpx[0:1,i:i+1], logpxi, inplace=True)
				logpx.append(logpxi)
				pass
			logpx = T.concatenate(logpx, 1)
			
		# Note: logpz is a row vector (one element per sample)
		logpz = 0
		for i in z:
			logpz += logpdfs.standard_normal(z[i]).sum(axis=0) # logp(z)
		
		# Note: logpw is a scalar
		logpw = 0
		for i in w:
			logpw += logpdfs.normal(w[i], 0, prior_sd).sum() # logp(w)
		
		return logpx, logpz, logpw

	def variables():
		# Define parameters 'w'
		w = {}
		for i in range(1, len(n_hidden)):
			w['w'+str(i)] = T.dmatrix('w'+str(i))
			w['b'+str(i)] = T.dmatrix('b'+str(i))
			w['logsd'+str(i)] = T.dmatrix('logsd'+str(i))
		w['wout'] = T.dmatrix('wout')
		w['bout'] = T.dmatrix('bout')
		
		# Define latent variables 'z'
		z = {}
		for i in range(len(n_hidden)):
			z['eps'+str(i)] = T.dmatrix('eps'+str(i))
		
		# Define observed variables 'x'
		x = {}
		x['x'] = T.dmatrix('x')
		
		return w, x, z
	
	# Confabulate hidden states 'z'
	def gen_xz(w, x, z):
		raise Exception('TODO')
		for i in range(len(n_hidden)):
			z['eps'+str(i)] = np.random.standard_normal(size=(n_hidden[i], n_batch))
		return x, z, {}

	def init_w(std):
		w = {}
		for i in range(1, len(n_hidden)):
			w['w'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], n_hidden[i-1]))
			w['b'+str(i)] = np.random.normal(0, std, size=(n_hidden[i], 1))
			w['logsd'+str(i)] = np.random.normal(0, 1, size=(n_hidden[i], 1))
		w['wout'] = np.random.normal(0, std, size=(n_output, n_hidden[-1]))
		w['bout'] = np.random.normal(0, std, size=(n_output, 1))
		return w
		
	return BNModel.BNModel(constr, variables, functions, init_w, gen_xz, n_batch)
