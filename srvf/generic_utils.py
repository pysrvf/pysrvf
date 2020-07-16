import numpy as np 
from scipy.interpolate import interp1d
import struct
import os

def inner_product_L2(u, v):
	'''
	Computes the standard inner product on L2
	Input:
	- u: An (n x T) matrix representation of the function u: D -> R^n
	- v: An (n x T) matrix representation of the function v: D -> R^n
	Outputs:
	<u,v> = int_[0,2pi] (u(t)v(t))_R^n dt
	'''

	n, T = np.shape(u)
	D = np.linspace(0, 2*np.pi, T, True)

	return np.trapz(np.sum(np.multiply(u, v), axis = 0), D)

def induced_norm_L2(u, D = None):
	'''
	Computes the norm induced by the standard L2 inner product
	- u: An (n x T) matrix representation of the function u: D -> R^n
	Outputs:
	||u|| = sqrt(<u,u>) = sqrt(int_[0,2pi] (u(t)u(t))_R^n dt)
	'''

	return np.sqrt(inner_product_L2(u, u))

def form_basis_normal_A(q):
	'''
	Returns vector field that forms the basis for normal space of cal(A)
	Input:
	- q: An (n x T) matrix representation of the SRVF q: [0,2pi] -> R^n
	Output:
	- del_g: An (n x n x T) array representing the vector field
	'''
	n, T = np.shape(q)
	e = np.eye(n)
	ev = []

	for i in range(n):
		ev.append( np.tile(e[:, i], (T, 1)).transpose())
		qnorm = np.zeros(T)

	qnorm = np.linalg.norm(q, 2, axis = 0)

	del_g = np.zeros((n, n, T))
	for i in range(n):
		tmp1 = np.tile((q[i, :] / qnorm), (n, 1))
		tmp2 = np.tile(qnorm, (n, 1))
		del_g[i] = tmp1*q + tmp2*ev[i]
	return del_g

def form_basis_D(d, T):
	''' 
	Returns the basis for the tangent space of diffeomorphisms
	Inputs:
	- d: A nonnegative integer
	- T: A nonnegative integer
	Ouputs:
	- V: A (2d x 10) matrix representing the basis
	'''
	x = np.reshape(np.linspace(0, 2*np.pi, T, True), (1, T))
	vec = np.reshape(np.arange(1, d+1), (1, d))
	x_d_arr = np.matmul(vec.T, x)
	V = np.vstack([np.cos(x_d_arr)/np.sqrt(np.pi), np.sin(x_d_arr)/np.sqrt(np.pi)])
	return V

def shiftF(p, tau):
	''' 
	Shifts the elements in the matrix p tau indices to the left 
	Inputs:
	- p: An (n x T) matrix
	- tau: An integer
	Outputs:
	An (n x T) matrix with the columns shifted to the left by tau units
	'''
	return np.roll(p, -np.abs(tau), axis = 1)

def regroup(q1, q2):
	'''
	Shifts the columns of q2 to the left x units where x is the
	shift amount giving the smallest distance between q1 and q2 shifted
	Inputs:
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	Outputs:
	- q2n: An (n x T) matrix which is q2 shifted to the left x units
	'''
	n, T = np.shape(q1)
	q2n = np.zeros_like(q2)
	E = np.zeros(T)

	for tau in range(T):
		q2n = shiftF(q2, tau)
		E[tau] = inner_product_L2(q1 - q2n, q1-q2n)

	idx = np.argmin(E)
	Emin = E[idx]
	q2n = shiftF(q2, idx)

	return q2n

def save_q_shapes(filename, qarr):
	'''
	Saves the given q-array into a binary file
	Inputs:
	- filename: The name of the file (with extenstion) to save to, 
				e.g., 'DPshapedata.dat'
	- qarr: An (N x n x T) array of q curves to write to binary file
	Outputs:
	0 if succeeded
	'''
	N, n, T = np.shape(qarr)

	with open(filename, 'wb') as fid:
		fid.write(struct.pack('i', N)) # Store number of shapes
		fid.write(struct.pack('i', n)) # Store number of dimensions
		fid.write(struct.pack('i', T)) # Store number of samples per shape

		for i in range(N):
			for j in range(n):
				fid.write(struct.pack('f'*T, *qarr[i,j,:]))

		fid.close()
	return 0

def load_gamma(filename):
	''' 
	Reads the binary file in filename and returns its content. Assumes this
	file is a diffeomorphism (T-dimensional vector) saved in binary format.
	- filename: The name of the file (with extenstion) to save to, 
				e.g., 'gamma.dat'
	Ouptus:
	- gamma: A T-dimensional vector
	'''

	gamma = []
	with open(filename, 'rb') as fid:
		gamma = np.fromfile(fid, '<f4', offset = 4) # Skip first byte containing T
		fid.close()

	return gamma

def group_action_by_gamma(q, gamma):
	'''
	Computes composition of q and gamma and normalizes by gradient
	Inputs:
	- q: An (n x T) matrix 
	- gamma: A T-dimensional vector representing the warp to apply to q
	Outputs:
	- qn: An (n x T) matrix representing the composition of q with gamma
		  normalized by the gradient
	'''
	n, T = np.shape(q)
	gamma_t = np.gradient(gamma, 2*np.pi/(T-1))
	D = np.linspace(0, 2*np.pi, T, True)

	q_composed_gamma = np.zeros_like(q)

	for i in range(n):
		f = interp1d(D, q[i,:], kind = 'nearest')
		q_composed_gamma[i,:] = f(gamma)

	sqrt_gamma_t = np.tile(np.sqrt(gamma_t), (n, 1))
	qn = np.multiply(q_composed_gamma, sqrt_gamma_t)

	return qn

def initialize_gamma_using_DP(q1, q2):
	'''
	Gets the warping function that warps q2 to q1 and the reparameterization of q2
	Inputs:
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	Outputs:
	- q2n: An (n x T) matrix representing the reparameterization of q2 (q2(gamma))
	- gamma: A T-dimensional vector of the differomorphism that warps q2 to q1
	'''

	# Create and save q-array
	qarr = np.array([q1, q2])
	save_q_shapes('DPshapedata.dat', qarr)

	# Call to get gamma
	os.system('./DP_Shape_Match_SRVF_nDim DPshapedata.dat gamma.dat')
