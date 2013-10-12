from IPython.parallel import Client

def map(r,func, args=None, modules=None):
	"""
	Before you run parallel.map, start your cluster (e.g. ipcluster start -n 4)
	
	map(r,func, args=None, modules=None):
	args=dict(arg0=arg0,...)
	modules='numpy, scipy'    
	
	examples:
	func= lambda x: numpy.random.rand()**2.
	z=parallel.map(r_[0:1000], func, modules='numpy, numpy.random')
	plot(z)
	
	A=ones((1000,1000));
	l=range(0,1000)
	func=lambda x : A[x,l]**2.
	z=parallel.map(r_[0:1000], func, dict(A=A, l=l))
	z=array(z)
	
	"""
	mec = Client()
	mec.clear()
	lview=mec.load_balanced_view()
	for k in mec.ids:
		mec[k].activate()
		if args is not None:
			mec[k].push(args)
		if modules is not None:
			mec[k].execute('import '+modules)
	z=lview.map(func, r)
	out=z.get()
	return out
