import numpy as np 
from scipy.integrate import cumtrapz
from pysrvf.generic_utils import inner_product_L2
from pysrvf.generic_utils import induced_norm_L2


def project_B(q, D = None):
	'''
	Projects given point in L2 onto unit Hilbert sphere in L2
	Inputs: 
	- q: An (n x T) matrix representation of the SRVF of function p: D -> R^n
	- D: A vector denoting the domain of the curve, i.e., p: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	An (n x T) matrix representation of q projected on the Hilbert sphere
	'''
	eps = 1e-8
	return q/(induced_norm_L2(q, D) + eps)

def curve_to_q(p, shape = None, D = None):
	''' 
	Given a curve p, gets the SRVF representation
	Inputs:
	- p: An (n x T) matrix representation of the function p: D -> R^n
	- shape: A vector denoting the size of p, i.e., [n, T]. If none given,
			 will determine shape of p. Used to speed up batch processing
	- D: A vector denoting the domain of the curve, i.e., p: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	An (n x T) matrix representation of the SRVF of p
	'''
	if shape is None:
		n, T = np.shape(p)
	else:
		n, T = shape

	if D is None:
		D = np.linspace(0, 2*np.pi, T, True)

	beta_dot = np.zeros((n,T))
	q = np.zeros((n,T))

	for i in range(n):
		beta_dot[i,:] = np.gradient(p[i,:], D)

	eps = np.finfo(float).eps
	for i in range(T):
		q[:,i] = beta_dot[:,i]/(np.sqrt(np.linalg.norm(beta_dot[:,i])) + eps)

	q = project_B(q, D)

	return q

def batch_curve_to_q(curves, D = None):
	''' 
	Given a collection of curves, gets their SRVF representation. Assumes that all
	matrix representations of the curves are of the same size. 
	Inputs:
	- curves: A (N x n x T) list of matrices.
	- D: A vector denoting the domain of the curves, i.e., p_i: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	A (N x n x T) list of matrices where the ith element is the SRVF representation
	of the ith curve in curves.
	'''
	N, n, T = np.shape(curves)

	return [curve_to_q(p_i, [n, T]) for p_i in curves]

def q_to_curve(q, shape = None, D = None):
	''' 
	Given an SRVF curve q, recovers original curve
	Inputs:
	- q: An (n x T) matrix representation of the SRVF q: D -> R^n
	- shape: A vector denoting the size of p, i.e., [n, T]. If none given,
			 will determine shape of p. Used to speed up batch processing
	- D: A vector denoting the domain of the curve, i.e., q: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	An (n x T) matrix representation of the original curve
	'''
	if shape is None:
		n, T = np.shape(q)
	else:
		n, T = shape

	if D is None:
		D = np.linspace(0, 2*np.pi, T, True)

	q_norms = np.linalg.norm(q, axis = 0)
	p = np.zeros((n,T))

	for i in range(n):
		p[i,:] = cumtrapz(np.multiply(q[i,:], q_norms), D, initial = 0)

	return p

def rescale_curves(p_curves, D = None):
	'''
	Rescales a collection of curves to be of unit length
	Inputs:
	- p_curves: An (N x n x T) array of curves
	Outputs:
	An (N x n x T) array of curves consisting of the rescaled curves in p_curves
	'''
	# Center curve
	p_curves = [np.transpose(np.transpose(p) - np.mean(p[:,:-1], axis = 1)) for p in p_curves]
	return [project_B(p, D) for p in p_curves]

def distance_function(u, v, D = None):
	'''
	Given two points on the SRVF Hilbert sphere, computes the distance given by
	<u - v, u - v>.
	Inputs:
	- u: An (n x T) matrix representation of the SRVF of u: D -> R^n
	- v: An (n x T) matrix representation of the SRVF of v: D -> R^n
	- D: A vector denoting the domain of the curve, i.e., u: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	A scalar, <u - v, u - v> = ||u - v||^2
	'''

	return induced_norm_L2(u - v, D)

def exponential_map(u, v, D = None):
	''' 
	Computes exp_u(v) 
	Inputs:
	- u: An (n x T) matrix representation of the SRVF of u: D -> R^n
	- v: An (n x T) matrix representation of the SRVF of v: D -> R^n
	- D: A vector denoting the domain of the curve, i.e., u: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	A scalar, exp_u(v) 
	'''
	eps = 1e-8
	v_norm = induced_norm_L2(v, D)

	return np.cos(v_norm)*u + (np.sin(v_norm)/(v_norm + eps))*v

def inverse_exponential_map(u, v, D = None):
	''' 
	Computes exp_u^{-1}(v) 
	Inputs:
	- u: An (n x T) matrix representation of the SRVF of u: D -> R^n
	- v: An (n x T) matrix representation of the SRVF of v: D -> R^n
	- D: A vector denoting the domain of the curve, i.e., u: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	A scalar, exp_u^{-1}(v) 
	'''
	uv_inner_prod = inner_product_L2(u, v, D) 
	eps = 1e-8

	if uv_inner_prod < -1:
		uv_inner_prod = -1 + eps
	elif uv_inner_prod > 1:
		uv_inner_prod = 1 - eps

	# theta = np.arccos(uv_inner_prod)

	# return theta/(np.sin(theta) + eps)*(v - uv_inner_prod*u)

	return np.arccos(uv_inner_prod)/np.sqrt(1 - uv_inner_prod**2)*(v - uv_inner_prod*u)
