import numpy as np
from generic_utils import *
from form_basis_utils import *
from scipy.integrate import trapz

def geodesic_Q(q1, q2, stp):
	'''
	Forms geodesic in Q from q1 (projected to C) to q2 (projected to C)
	Inputs:
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	- stp: Number of points (-1) to take in geodesic
	Outputs:
	- alpha_new: An ((stp+1) x n x T) array representing the geodesic
	'''
	n, T = np.shape(q1)
	ip_L2 = inner_product_L2(q1, q2)
	if ip_L2 > 1:
		ip_L2 = 1

	if ip_L2 < -1:
		ip_L2 = -1
	
	theta = np.arccos(ip_L2)
	alpha_new = np.zeros((stp + 1, n, T))
	eps = np.spacing(1)
	
	for tau in range(stp+1):
		t = tau/stp
		if theta < 1e-4:
			alpha_new[tau] = q1
		else:
			alpha_new[tau] = projectC((np.sin(theta-t*theta)*q1 + np.sin(t*theta)*q2)/(np.sin(theta)+eps))

	return alpha_new

def dAlpha_dt(alpha):
	'''
	Computes the derivative of the given alpha
	Inputs:
	- alpha: A (k x n x T) array representing a geodesic
	Outputs:
	- alpha_t: A (k x n x T) array representing the derivative of alpha
	'''
	k, n, T = np.shape(alpha)
	stp = k-1
	alpha_t = np.zeros_like(alpha)

	for tau in np.arange(1, k):
		alpha_t[tau] = stp*(alpha[tau] - alpha[tau-1])
		alpha_t[tau] = project_tangent(alpha_t[tau], alpha[tau])

	return alpha_t

def path_length(alpha_t):
	'''
	Computes the length of the given geodesic derivative
	Inputs:
	- alpha_t: A (k x n x T) array representing the derivative of alpha
	Outputs:
	- L: A nonnegative scalar representing the length of alpha_t
	'''
	k, _, _ = np.shape(alpha_t)
	s = np.linspace(0, 1, k)
	v_norm = np.zeros(k)
	
	for i in range(k):
		v_norm[i] = induced_norm_L2(alpha_t[i])

	L = trapz(v_norm, s)

	return L

def palais_inner_product(v1, v2):
	'''
	Computes ... inner product
	Inputs: 
	- v1: A (k x n x T) array
	- v2: A (k x n x T) array
	Outputs:
	- val: A scalar representing the ... inner product of v1 and v2
	'''
	k, n, T = np.shape(v1)
	s = np.linspace(0, 1, k)
	v_inner = np.zeros(k)

	for i in range(k):
		v_inner[i] = inner_product_L2(v1[i], v2[i])

	val = trapz(v_inner, s)

	return val

def parallel_transport_C(w, q1, q2):
	'''
	Parallel transports vector w from alpha1 in T_alpha1(C) to
	alpha2 in T_alpha2(C)
	Inputs:
	- w: An (n x T) matrix
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	Outputs:
	- w_new: An (n x T) matrix representing the transported w
	'''
	w_norm = induced_norm_L2(w)
	w_new = w

	if w_norm > 1e-4:
		w_new = project_tangent(w, q2)
		w_new = w_norm*w_new/induced_norm_L2(w_new)

	return w_new

def back_parallel_transport_C(w_final, alpha):
	'''
	Computes ...
	Inputs:
	- w: An (n x T) matrix
	- alpha: A (k x n x T) array representing a geodesic
	Outputs:
	- w_tilde: An (n x T) matrix
	- t_w_tilde: An (n x T) matrix 
	'''
	k, n , T = np.shape(alpha)
	stp = k-1

	w_tilde = np.zeros_like(alpha)
	t_w_tilde = np.zeros_like(alpha)
	w_tilde[-1] = w_final
	t_w_tilde[-1] = w_final

	for tau in np.arange(k-2, -1, -1):
		w_tilde[tau] = parallel_transport_C(w_tilde[tau+1], alpha[tau+1], alpha[tau])
		t_w_tilde[tau] = tau/stp*w_tilde[tau]

	return w_tilde, t_w_tilde

def cov_int_alpha_t(alpha, alpha_t):
	'''
	Computes the covariant integration of alpha_t
	Inputs:
	- alpha: A (k x n x T) array 
	- alpha_t: A (k x n x T) array representing the derivative of alpha
	Outputs:
	- w: A (k x n x T) array representing the covariant integration
	'''
	k, n, T = np.shape(alpha_t)
	stp = k-1
	w = np.zeros_like(alpha_t)

	for tau in np.arange(1, k):
		w_prev = parallel_transport_C(w[tau-1], alpha[tau-1], alpha[tau])
		w[tau] = w_prev + alpha_t[tau]/stp
		w[tau] = project_tangent(w[tau], alpha[tau])

	return w

def cov_derivative(w, q):
	'''
	Computes the covariant derivative of w
	Inputs:
	- w: A (k x n x T) array
	- q: A (k x n x T) array
	Outputs:
	- w_t_cov: A (k x n x T) array
	'''
	k, n, T = np.shape(q)
	stp = k-1

	w_t = np.zeros_like(w)
	w_t_cov = np.zeros_like(w)

	for tau in np.arange(1, k):
		w_t[tau] = stp*(w[tau] - w[tau-1])
		w_t_cov[tau] = project_tangent(w_t[tau], q[tau])

	return w_t_cov

def geodesic_sphere(x_init, g, dt):
	'''
	Computes a geodesic on the sphere at starting point x_init in direction g
	Inputs:
	- x_init: An (n x T) matrix
	- g: An (n x T) matrix
	- dt: A scalar
	Outputs:
	- X: An (n x T) matrix
	'''
	g_norm = induced_norm_L2(g)
	X = np.cos(dt*g_norm)*x_init + np.sin(dt*g_norm)*g/g_norm

	return X


def path_update(alpha, v, dt):
	'''
	Updates the given alpha along direction v
	Inputs:
	- alpha: A (k x n x T) array representing a geodesic
	- v: A (k x n x T) array
	- dt: A scalar
	Outputs:
	- alpha_new: A (k x n x T) array representing the updated geodesic
	'''
	k, n, T = np.shape(alpha)
	stp = k-1
	alpha_new = np.zeros_like(alpha)

	for tau in range(k):
		if induced_norm_L2(v[tau]) < 1e-4:
			alpha_new[tau] = alpha[tau]
		else:
			alpha_new[tau] = projectC(geodesic_sphere(alpha[tau], v[tau], dt))

	return alpha_new
