import numpy as np 
import struct
import os
import warnings
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from dpmatchsrvf import dpmatch

def inner_product_L2(u, v):
	'''
	Computes the standard inner product on L2
	Input:
	- u: An (n x T) matrix representation of the function u: D -> R^n
	- v: An (n x T) matrix representation of the function v: D -> R^n
	Outputs:
	<u,v> = int_[0,2pi] (u(t)v(t))_R^n dt
	'''

	_, T = np.shape(u)

	return np.trapz(np.sum(np.multiply(u, v), axis = 0), dx = 2*np.pi/(T-1))

def induced_norm_L2(u):
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
		ev.append(np.tile(np.reshape(e[:, i], (n,1)), (1, T)))

	qnorm = np.linalg.norm(q, 2, axis = 0)

	del_g = np.zeros((n, n, T))
	for i in range(n):
		tmp1 = np.tile(np.divide(q[i,:], qnorm), (n, 1))
		tmp2 = np.tile(qnorm, (n, 1))
		del_g[i] = np.multiply(tmp1, q) + np.multiply(tmp2, ev[i])
	return del_g

def projectC(q):
    ''' 
    Projects the given SRVF curve onto the space of closed curves
    Inputs:
    - q: An (n x T) matrix representing the SRVF representation of a curve
    Outputs:
    - qnew: An (n x T) matrix representing the SRVF of a closed curve
    '''
    n, T = np.shape(q)
    dt = 0.3
    epsilon = (1.0/60.0)*(2.0*np.pi/T)

    k = 0
    res = np.ones((1,n))
    J = np.zeros((n, n))
    
    eps = np.spacing(1)
    qnew = q/(induced_norm_L2(q) + eps)

    while(np.linalg.norm(res, 2) > epsilon):
        if k > 300:
            warnings.warn('Shape failed to project. Geodesics will be incorrect.')
            break

        # Compute Jacobian
        for i in range(n):
            for j in range(n):
                J[i, j] = 3*np.trapz(np.multiply(qnew[i, :], qnew[j, :]), dx = 2*np.pi/(T-1))
        J += np.eye(n)

        qnorm = np.linalg.norm(qnew, 2, axis = 0)

        # Compute residue
        res = [-np.trapz(np.multiply(qnew[i,:], qnorm), dx = 2*np.pi/(T-1)) for i in range(n)]

        if np.linalg.norm(res, 2) < epsilon:
            break

        J_cond = np.linalg.cond(J)

        if np.isnan(J_cond) or np.isinf(J_cond) or (J_cond < 0.1):
            warnings.warn('Projection may not be accurate.')
            return q/(induced_norm_L2(q) + eps)
        else:
            x = np.linalg.solve(J, res)
            del_G = form_basis_normal_A(qnew)
            temp = 0
            for i in range(n):
                temp += x[i]*del_G[i]*dt
            qnew += temp
            k += 1

    qnew = qnew/induced_norm_L2(qnew)
    
    return qnew

def project_B(q):
	'''
	Projects given point in L2 onto unit Hilbert sphere in L2
	Inputs: 
	- q: An (n x T) matrix representation of the SRVF of function p: D -> R^n
	- D: A vector denoting the domain of the curve, i.e., p: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	An (n x T) matrix representation of q projected on the Hilbert sphere
	'''
	
	return q/induced_norm_L2(q) 
	
def curve_to_q(p, closed = False, shape = None):
	''' 
	Given a curve p, gets the SRVF representation
	Inputs:
	- p: An (n x T) matrix representation of the function p: D -> R^n
	- closed: A boolean indicating whether the passed curve is closed
	- shape: A vector denoting the size of p, i.e., [n, T]. If none given,
			 will determine shape of p
	Outputs:
	An (n x T) matrix representation of the SRVF of p
	'''
	if shape is None:
		n, T = np.shape(p)
	else:
		n, T = shape

	beta_dot = np.zeros((n,T))
	q = np.zeros((n,T))

	for i in range(n):
		beta_dot[i,:] = np.gradient(p[i,:], 2*np.pi/(T-1))

	eps = np.finfo(float).eps
	for i in range(T):
		q[:,i] = beta_dot[:,i]/(np.sqrt(np.linalg.norm(beta_dot[:,i])) + eps)

	if not closed:
		q = project_B(q)
	else:
		q = projectC(q)

	return q

def is_curve_closed(curve):
	'''
	Determines if the given curve is closed or open
	Inputs:
	- curve: An (n x T) matrix representing a curve from [0, 2pi] -> R^n
	Outputs:
	A boolean. True if the curve is closed, false otherwise
	'''
	n, T = np.shape(curve)
	if n == 1:
		return False
		
	return np.sum(np.abs(curve[:,0] - curve[:,-1])) <= 1e-15

def batch_curve_to_q(curves):
	''' 
	Given a collection of curves, gets their SRVF representation. Assumes that all
	matrix representations of the curves are of the same size. 
	Inputs:
	- curves: A (N x n x T) list of matrices.
	Outputs:
	A (N x n x T) list of matrices where the ith element is the SRVF representation
	of the ith curve in curves.
	'''
	N, n, T = np.shape(curves)

	# Determine if first curve is closed. All other curves will be labeled accordingly
	is_closed = is_curve_closed(curves[0])
	if is_closed:
		print('Closed curves detected. {} total curves.'.format(N))
	else:
		print('Open curves detected. {} total curves.'.format(N))

	return [curve_to_q(p_i, is_closed, [n, T]) for p_i in curves], is_closed

def q_to_curve(q):
	''' 
	Given an SRVF curve q, recovers original curve
	Inputs:
	- q: An (n x T) matrix representation of the SRVF q: D -> R^n
	Outputs:
	An (n x T) matrix representation of the original curve
	'''
	
	n, T = np.shape(q)

	q_norms = np.linalg.norm(q, axis = 0)
	p = np.zeros((n,T))

	for i in range(n):
		p[i,:] = cumtrapz(np.multiply(q[i,:], q_norms), dx = 2*np.pi/(T-1), initial = 0)

	return p

def batch_q_to_curve(srvfs):
	''' 
	Given a collection of SRVF, gets their original representation. Assumes that all
	matrix representations of the curves are of the same size. 
	Inputs:
	- srvfs: A (N x n x T) list of matrices.
	Outputs:
	A (N x n x T) list of matrices where the ith element is the original representation
	of the ith srvf in srvfs.
	'''
	N, n, T = np.shape(srvfs)

	# Determine if first curve is closed. All other curves will be labeled accordingly
	is_closed = is_curve_closed(srvfs[0])
	if is_closed:
		print('Closed srvfs detected. {} total curves.'.format(N))
	else:
		print('Open srvfs detected. {} total curves.'.format(N))

        #Make sure to add additional parameters to match curve_to_q
	return [q_to_curve(q_i) for q_i in srvfs], is_closed
	
def reparameterize_curve_gamma(curve, gamma):
	'''
	Applies the warping function gamma to the given curve
	Inputs:
	- curve: An (n x T) matrix representing the curve
	- gamma: A T-dimensional vector representing the warping function
	Outputs:
	- An (n x T) matrix representation of the warped curve
	'''

	n, T = np.shape(curve)
	pn = np.zeros_like(curve)

	for i in range(n):
		f = interp1d(np.linspace(0, 2*np.pi, T), curve[i,:], kind = 'linear')
		pn[i,:] = f(gamma)

	return pn

def find_best_rotation(q1, q2):
	'''
	Solves Procrusted problem to find optimal rotation
	Inputs:
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	Outputs:
	- q2n: An (n x T) matrix representing the rotated q2
	- R: An (n x n) matrix representing the rotation matrix
	'''
	n, T = np.shape(q1)
	A = np.matmul(q1, q2.T)
	[U, S, V] = np.linalg.svd(A)
	V = V.T

	S = np.eye(n)
	if (np.abs(np.linalg.det(U)*np.linalg.det(V) - 1) > 10*np.spacing(1)):
		S[:,-1] = -S[:,-1]

	R = np.matmul(U, np.matmul(S, V.T))
	q2n = np.matmul(R, q2)

	return q2n, R

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

def find_rotation_and_seed_unique(q1, q2):
	'''
	Finds locally optimal rotatino and seed point for q2 wrt q1
	Inputs:
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	Outputs:
	- q2new: An (n x T) matrix representing the rotated q2
	'''
	n, T = np.shape(q1)
	L2_norm = []
	R_arr = []
	for ctr in range(T+1):
		q2n = shiftF(q2, ctr)
		q2new, R = find_best_rotation(q1, q2n)
		L2_norm.append(induced_norm_L2(q1 - q2new)**2)
		R_arr.append(R)

	L2_norm_amin = np.argmin(L2_norm)
	L2_norm_min = L2_norm[L2_norm_amin]
	q2new = shiftF(q2, L2_norm_amin)
	q2new = np.matmul(R_arr[L2_norm_amin], q2new)

	return q2new

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

	# gamma = []
	# with open(filename, 'rb') as fid:
	# 	# Skip first byte containing T. Assume little-endian
	# 	gamma = np.fromfile(fid, '<f4', offset = 4) 
	# 	fid.close()

	f = open(filename, 'rb')
	T = struct.unpack('i', f.read(4))[0]
	gamma = np.array(struct.unpack('f'*T,  f.read()))
	f.close()

	return gamma

def group_action_by_gamma(q, gamma, is_closed):
	'''
	Computes composition of q and gamma and normalizes by gradient
	Inputs:
	- q: An (n x T) matrix 
	- gamma: A T-dimensional vector representing the warp to apply to q
	- is_closed: A boolean indicating whether the original curves are closed
	Outputs:
	- qn: An (n x T) matrix representing the composition of q with gamma
		  normalized by the gradient
	'''
	n, T = np.shape(q)
	D = np.linspace(0, 2*np.pi, T, True)

	if is_closed:
		gamma_t = np.gradient(gamma, 2*np.pi/(T-1))

		q_composed_gamma = np.zeros_like(q)

		for i in range(n):
			f = interp1d(D, q[i,:], kind = 'nearest')
			q_composed_gamma[i,:] = f(gamma)

	else:
		gamma_t = np.gradient(gamma, 2*np.pi/T) # CHECK THIS
		f = interp1d(D, q, kind = 'linear', fill_value = 'extrapolate')
		q_composed_gamma = f(gamma)

	sqrt_gamma_t = np.tile(np.sqrt(gamma_t), (n, 1))
	qn = np.multiply(q_composed_gamma, sqrt_gamma_t)

	return qn

def initialize_gamma_using_DP(q1, q2, is_closed):
	'''
	Gets the warping function that warps q2 to q1 and the reparameterization of q2
	Inputs:
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	- is_closed: A boolean indicating whether the original curves are closed
	Outputs:
	- q2n: An (n x T) matrix representing the reparameterization of q2 (q2(gamma))
	- gamma: A T-dimensional vector of the diffeomorphism that warps q2 to q1
	'''

	# Create and save q-array
	# qarr = np.array([q1, q2])
	# save_q_shapes('DPshapedata.dat', qarr)

	# Call to get gamma
	# os.system('./DP_Shape_Match_SRVF_nDim DPshapedata.dat gamma.dat')
	# os.system('./DPShapeMatch DPshapedata.dat gamma.dat')

	gamma = dpmatch().match(q1, q2)

	# gamma = load_gamma('gamma.dat')
	gamma = 2*np.pi*gamma/np.max(gamma)
	q2n = group_action_by_gamma(q2, gamma, is_closed)

	return q2n, gamma

def initialize_gamma_using_DP_symm(q1, q2, is_closed):
	'''
	Gets the warping function that warps q2 to q1 and the reparameterization of q2
	Inputs:
	- q1: An (n x T) matrix
	- q2: An (n x T) matrix
	- is_closed: A boolean indicating whether the original curves are closed
	Outputs:
	- q2n: An (n x T) matrix representing the reparameterization of q2 (q2(gamma))
	- gamma: A T-dimensional vector of the differomorphism that warps q2 to q1
	'''

	# Create and save q-array
	qarr = np.array([q1, q2])
	save_q_shapes('DPshapedata.dat', qarr)

	# Call to get gamma
	# TODO: Replace DP_Shape_Match_SRVF_nDim with DP_matchSymm when available
	os.system('./DP_Shape_Match_SRVF_nDim DPshapedata.dat gamma.dat')

	gamma = load_gamma('gamma.dat')
	gamma = 2*np.pi*gamma/np.max(gamma)
	q2n = group_action_by_gamma(q2, gamma, is_closed)

	return q2n, gamma

def estimate_gamma(q, is_closed):
	'''
	Estimate warping function given curve
	Inputs:
	- q: An (n x T) matrix
	- is_closed: A boolean indicating whether the original curves are closed
	Outputs:
	- gamma: A T-dimensional vector representing a diffeomorphism
	'''

	p = q_to_curve(q)
	n, T = np.shape(p)

	if is_closed:
		# Evaluate arc-length formula
		pdiff = np.diff(p, 1)
		ds = T*np.sqrt(np.sum(np.square(pdiff), axis = 0))
		ds_cumsum = np.cumsum(ds)
		gamma = 2*np.pi*ds_cumsum/np.max(ds_cumsum)
	else:
		if n == 1:
			pgrad = np.gradient(p[0,:], 2*np.pi/T)
			ds = T*np.abs(pgrad)
		else:
			pgrad = np.gradient(p, 2*np.pi/T)
			ds = T*np.sqrt(np.sum(np.square(pgrad), axis = 0))
		cum_sum = cumtrapz(ds, dx = 2*np.pi/T, initial = 0)
		gamma = 2*np.pi*cum_sum/np.max(cum_sum)

	return gamma
