import numpy as np 

def inner_product_L2(u, v, D = None):
	'''
	Computes the standard inner product on L2
	Input:
	- u: An (n x T) matrix representation of the function u: D -> R^n
	- v: An (n x T) matrix representation of the function v: D -> R^n
	- D: A vector of length T denoting the points at which u and v are sampled.
		 If none, will discretize [0, 2pi] into T points.
	Outputs:
	<u,v> = int_D (u(t)v(t))_R^n dt
	'''
	n, T = np.shape(u)

	if D is None:
		D = np.linspace(0, 2*np.pi, T, True)

	return np.trapz(np.sum(np.multiply(u, v), axis = 0), D)

def induced_norm_L2(u, D = None):
	'''
	Computes the norm induced by the standard L2 inner product
	- u: An (n x T) matrix representation of the function u: D -> R^n
	- D: A vector of length T denoting the points at which u and v are sampled.
		 If none, will discretize [0, 2pi] into T points.
	Outputs:
	||u|| = sqrt(<u,u>) = sqrt(int_D (u(t)u(t))_R^n dt)
	'''

	return np.sqrt(inner_product_L2(u, u, D))