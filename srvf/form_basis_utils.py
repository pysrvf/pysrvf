import numpy as np
import warnings
from generic_utils import induced_norm_L2
from generic_utils import inner_product_L2

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

def projectC(q):
    ''' 
    Projects the given SRVF curve onto the space of closed curves
    Inputs:
    - q: An (n x T) matrix representing the SRVF representation of a curve
    Outputs:
    - qnew: An (n x T) matrix representing the SRVF of a closed curve
    '''

    n, T = np.shape(q)
    D = np.linspace(0, 2*np.pi, T)
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
                J[i, j] = 3*np.trapz(np.multiply(qnew[i, :], qnew[j, :]), D)
        J += np.eye(n)

        qnorm = np.linalg.norm(qnew, 2, axis = 0)

        # Compute residue
        res = [-np.trapz(np.multiply(qnew[i,:], qnorm), D) for i in range(n)]

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

def gram_schmidt(X):
	'''
	Applies Gram-schmidt orthonormalization to X with L2 inner product
	Inputs:
	- X: An (m x n) matrix
	Outputs:
	- Y: An (m x n) matrix with orthonormal columns (wrt L2 inner product)
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

def project_tangent(f, q):
	''' 
	Projects tangent vector f 
	Inputs:
	- f: An (n x T) matrix representing tangent vector at q
	- q: An (n x T) matrix 
	Outputs:
	- fnew: Projection of f onto tangent space of C at q
	'''
	n, T = np.shape(q)

	# Project w in T_q(C(B))
	w = f - np.multiply(inner_product_L2(f, q), q)
	e = np.eye(n)

	# Get basis for normal space of C(A)
	g = form_basis_normal_A(q)

	Ev = gram_schmidt(g)

	s = np.zeros_like(Ev[0])

	for i in range(n):
		s += inner_product_L2(w, Ev[i])*Ev[i]

	fnew = w - s

	return fnew
	