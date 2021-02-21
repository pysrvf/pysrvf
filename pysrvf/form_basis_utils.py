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
    x = np.linspace(0, 2*np.pi, T)
    xdarray = np.arange(1,d+1)
    xdarray = np.outer(xdarray,x)
    V_cos = np.cos(xdarray) / np.sqrt(np.pi)
    V_sin = np.sin(xdarray) / np.sqrt(np.pi)
    V = np.concatenate((V_cos,V_sin))
    x = np.reshape(np.linspace(0, 2*np.pi, T, True), (1, T))
    return V

def form_basis_D_q(V, q):
    '''
    Returns ...
    Inputs:
    - V: A ((2*d) x T) matrix representing basis for tangent space of diffeomorphisms
    - q: An (n x T) matrix representation of the SRVF q: [0,2pi] -> R^n
    Ouputs:
    - D_q: A ((2*d) x n x T) array
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
    - D_q: A ((2*d) x n x T) array
    Ouputs:
    - u_proj: An (n x T) matrix of projection of u
    - a: A (1 x (2*d)) vector representing Fourier coefficients
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


def form_basis_L2_R3(d, T):
    '''
    Returns Schauder basis for L_2(R^3)
    Note basis elements will be 6 x d
    '''

    x = np.linspace(0, 1, T, True)
    sqrt_2 = np.sqrt(2)
    constB = np.zeros((3, 3, T))

    constB[0] = np.array([np.sqrt(2) * np.ones(T), np.zeros(T), np.zeros(T)])
    constB[1] = np.array([np.zeros(T), np.sqrt(2) * np.ones(T), np.zeros(T)])
    constB[2] = np.array([np.zeros(T), np.zeros(T), np.sqrt(2) * np.ones(T)])

    B = np.zeros((6*d, 3, T))
    k = 0
    for j in np.arange(1, d+1):
        B[0 + 6*k] = np.array([np.sqrt(2) * np.cos(2 * np.pi * j * x), np.zeros(T), np.zeros(T)])
        B[1 + 6*k] = np.array([np.zeros(T), np.sqrt(2) * np.cos(2 * np.pi * j * x), np.zeros(T)])
        B[2 + 6*k] = np.array([np.zeros(T), np.zeros(T), np.sqrt(2) * np.cos(2 * np.pi * j * x)])
        B[3 + 6*k] = np.array([np.sqrt(2) * np.sin(2 * np.pi * j * x), np.zeros(T), np.zeros(T)])
        B[4 + 6*k] = np.array([np.zeros(T), np.sqrt(2) * np.sin(2 * np.pi * j * x), np.zeros(T)])
        B[5 + 6*k] = np.array([np.zeros(T), np.zeros(T), np.sqrt(2) * np.sin(2 * np.pi * j * x)])
        k = k + 1

    B = np.concatenate((constB, B))

    return B


def form_basis_L2_R2(d, T):
    '''
    Returns Schauder basis for L_2(R^2)
    Note basis elements will be 4 x d + 2
    '''

    x = np.linspace(0, 1, T, True)
    sqrt_2 = np.sqrt(2)
    constB = np.zeros((2, 2, T))

    constB[0] = np.array([np.sqrt(2) * np.ones(T), np.zeros(T)])
    constB[1] = np.array([np.zeros(T), np.sqrt(2) * np.ones(T)])

    B = np.zeros((4*d, 2, T))
    k = 0
    for j in np.arange(1, d+1):
        B[0 + 4*k] = np.array([np.sqrt(2) * np.cos(2 * np.pi * j * x), np.zeros(T)])
        B[1 + 4*k] = np.array([np.zeros(T), np.sqrt(2) * np.cos(2 * np.pi * j * x)])
        B[2 + 4*k] = np.array([np.sqrt(2) * np.sin(2 * np.pi * j * x), np.zeros(T)])
        B[3 + 4*k] = np.array([np.zeros(T), np.sqrt(2) * np.sin(2 * np.pi * j * x)])
        k = k + 1

    B = np.concatenate((constB, B))

    return B


def form_basis_of_tangent_space_of_S_at_q(Bnew, G_O_q):

    # T_q(S) = T_q(C) + T_q(O_q)^{\perp}
    # S in this case references closed orbits of C^o
    # S = {[q] | q in C^o}
    # Subtract the projection of basis of T_q(C) onto T_q(O_q) from itself
    # i.e. basis(T_q(C)) - <basis(T_q(C)), basis(T_q(O_q))> * basis(T_q(O_q))

    Gnew = Bnew.copy()
    for jj in np.arange(0, np.shape(Bnew)[0]):
        tmp = 0
        for kk in np.arange(0, np.shape(G_O_q)[0]):
            tmp = tmp + inner_product_L2(Bnew[jj], G_O_q[kk])*G_O_q[kk];
        # tmp calculates projections of vectors in T_q(C) onto T_q(O_q)
        # by iteratively summing up over the projections along the
        # orthonormal basis of T_q(O_q)
        Gnew[jj] = Bnew[jj] - tmp

    return Gnew


def project_to_basis(alpha_t_array, Y):

    V = np.zeros(alpha_t_array.shape) # orginally Y.shape
    #A = np.zeros((Y.shape[0], Y.shape[0]))
    A = np.zeros((alpha_t_array.shape[0], Y.shape[0]))
    d,n,T = Y.shape
    #n, T = np.shape(Y[0])
    for ii in np.arange(0, alpha_t_array.shape[0]):
    #for ii in np.arange(0, Y.shape[0]):
        V[ii] = np.zeros((n,T))
        for jj in np.arange(0, Y.shape[0]):
            A[ii, jj] = inner_product_L2(alpha_t_array[ii], Y[jj])
            V[ii] = V[ii] + A[ii, jj] * Y[jj]

    return A, V

def form_basis_O_q(B,q):
    d = len(B)
    n,T = q.shape

    # Form basis for L2(I, R)
    V = form_basis_D(d,T)
    if n == 2:
        R0 = np.array([[1, 0], [0, 1]])
        R1 = np.array([[0, 1], [-1, 0]])
        G = np.zeros((n,n,T))
        G[0] = R0 @ q
        G[1] = R1 @ q

    if n == 3:
        R0 = np.array([[0,1,0], [-1,0,0], [0,0,0]])
        R1 = np.array([[0,0,1], [0,0,0], [-1,0,0]])
        R2 = np.array([[0,0,0], [0,0,1], [0,-1,0]])

        G = np.zeros((n,n,T))
        G[0] = R0 @ q
        G[1] = R1 @ q
        G[2] = R2 @ q

    # Calculate derivatives of q
    qdiff = np.zeros(q.shape)
    for i in range(0,n):
        qdiff[i,:] = np.gradient(q[i,:],2*np.pi /(T-1))

    # Calculate the derivative of V
    Vdiff = np.zeros(V.shape)
    for i in range(0,d):
        Vdiff[i,:] = np.gradient(V[i,:], 2*np.pi / (T-1))

    D_q = np.zeros((d,n,T))
    for i in range(0,d):
        tmp1 = np.tile(V[i,:],(n,1))
        tmp2 = np.tile(Vdiff[i,:],(n,1))
        D_q[i] = np.multiply(qdiff,tmp1) + (1/2) * np.multiply(q,tmp2)

    O_q = np.concatenate((G,D_q))

    return O_q


def recon_from_basis(Aproj, Y):
    d,n,T = Y.shape
    N, _ = Aproj.shape
    V = np.zeros((N, n, T))
    for ii in np.arange(0, Aproj.shape[0]):
        V[ii] = np.zeros((n,T))
        for jj in np.arange(0, Y.shape[0]):
            V[ii] = V[ii] + Aproj[ii, jj] * Y[jj]
    return V
