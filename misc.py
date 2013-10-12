import numpy as np

def random_orthogonal(dim, special=True):
	if dim == 1:
		if np.random.uniform() < 0.5:
			return np.ones((1,1))
		return -np.ones((1,1))
	P = np.random.randn(dim, dim)
	while np.linalg.matrix_rank(P) != dim:
		P = np.random.randn(dim, dim)
	U, S, V = np.linalg.svd(P)
	P = np.dot(U, V)
	if special:
		# Make sure det(P) == 1
		if np.linalg.det(P) < 0:
			P[:, [0, 1]] = P[:, [1, 0]]
	return P
