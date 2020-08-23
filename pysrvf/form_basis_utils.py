import numpy as np
import warnings
from pysrvf.generic_utils import *


def form_basis_D(d, T):
	''' 
	Returns the basis for the tangent space of diffeomorphisms
	Inputs:
	- d: A nonnegative integer (number of Fourier coefficients divided by 2)
	- T: A nonnegative integer
	Ouputs:
	- V: A (2d x T) matrix representing the basis
	'''
	x = np.reshape(np.linspace(0, 2*np.pi, T, True), (1, T))
	vec = np.reshape(np.arange(1, d+1), (1, d))
	x_d_arr = np.matmul(vec.T, x)
	V = np.vstack([np.cos(x_d_arr)/np.sqrt(np.pi), np.sin(x_d_arr)/np.sqrt(np.pi)])
	return V

def form_basis_D_q(V, q):
	''' 
	Returns ... 
	Inputs:
	- V: A (d x T) matrix representing basis for tangent space of diffeomorphisms
	- q: An (n x T) matrix representation of the SRVF q: [0,2pi] -> R^n
	Ouputs:
	- D_q: A (d x n x T) array
	'''
	d, _ = np.shape(V)
	n, T = np.shape(q)

	qdiff = np.array([np.gradient(q[i,:], 2*np.pi/(T-1)) for i in range(n)])
	Vdiff = np.array([np.gradient(V[i,:], 2*np.pi/(T-1)) for i in range(d)])

	D_q = np.zeros((d, n, T))

	for i in range(d):
		tmp1 = np.tile(V[i,:], (n,1))
		tmp2 = np.tile(Vdiff[i,:], (n,1))
		D_q[i] = np.multiply(qdiff, tmp1) + 0.5*np.multiply(q, tmp2)

	return D_q

def project_tgt_D_q(u, D_q):
	''' 
	Returns ... 
	Inputs:
	- u: An (n x T) matrix
	- D_q: A (d x n x T) array
	Ouputs:
	- u_proj: An (n x T) matrix of projection of u 
	- a: A d-dimensional vector representing Fourier coefficients
	'''
	d, n, T = np.shape(D_q)

	u_proj = np.zeros((n, T))
	a = np.zeros(d)

	for i in range(d):
		a[i] = inner_product_L2(u, D_q[i])
		u_proj += a[i]*D_q[i]

	return u_proj, a

def gram_schmidt(X):
	'''
	Applies Gram-schmidt orthonormalization to X with L2 inner product
	Inputs:
	- X: An (N x n x T) matrix
	Outputs:
	- Y: An (N x n x T) matrix with orthonormal columns (wrt L2 inner product)
	'''
	epsilon = 5e-6
	N, n, T = np.shape(X)

	i = 0
	r = 0
	Y = np.zeros_like(X)
	Y[0] = X[0]

	while (i < N):
		temp_vec = 0
		for j in range(i):
			temp_vec += inner_product_L2(Y[j], X[r])*Y[j]
		Y[i] = X[r] - temp_vec
		temp = inner_product_L2(Y[i], Y[i])
		if temp > epsilon:
			Y[i] /= np.sqrt(temp)
			i += 1
			r += 1
		else:
			if r < i:
				r += 1
			else:
				break

	return Y

def project_tangent(f, q, is_closed):
	''' 
	Projects tangent vector f 
	Inputs:
	- f: An (n x T) matrix representing tangent vector at q
	- q: An (n x T) matrix 
	- is_closed: A boolean indicating whether the original curves are closed
	Outputs:
	- fnew: Projection of f onto tangent space of C at q
	'''
	n, T = np.shape(q)

	# Project w in T_q(C(B))
	w = f - inner_product_L2(f, q)*q

	if not is_closed:
		return w

	# Get basis for normal space of C(A)
	g = form_basis_normal_A(q)

	Ev = gram_schmidt(g)

	s = np.sum(inner_product_L2(w, Ev_i)*Ev_i for Ev_i in Ev)

	fnew = w - s

	return fnew
	