from generic_utils import induced_norm_L2
import numpy as np

def form_basis_normal_a(q):
    n, T = np.shape(q)

    e = np.eye(n)
    ev = []
    for i in range(n):
        ev.append( np.tile(e[:, i], (T, 1)).transpose())
        qnorm = np.zeros(T)
    for i in range(T):
        qnorm[i] = np.linalg.norm(q[:, i], ord=2)
    del_g = {}
    for i in range(n):
        tmp1 = np.tile((q[i, :] / qnorm), (n, 1))
        tmp2 = np.tile(qnorm, (n, 1))
        del_g[i] = tmp1*q + tmp2*ev[i]
    return del_g

def projectC(q, D = None):
    ''' 
    Projects the given SRVF curve onto the space of closed curves
    Inputs:
    - q: An (n x T) matrix representing the SRVF representation of a curve
    - D: A vector denoting the domain of the curves, i.e., q: D -> R^n. If none, will
         take D = [0, 2pi] discretized into T points
    Outputs:
    An (n x T) matrix representing the SRVF of a closed curve
    '''

    n, T = np.shape(q)

    if D is None:
        D = np.linspace(0, 2*np.pi, T, True)

    dt = 0.3
    epsilon = (1.0/60.0)*(2.0*np.pi/T) # May need to update if D \neq [0, 2pi]

    itr = 0
    res = np.ones((1, n))
    J = np.zeros((n, n))
    
    eps = np.spacing(1)
    qnew = q
    qnew = q/(np.sqrt(induced_norm_L2(q)) + eps)
    C = []
    while np.linalg.norm(res, ord=2) > epsilon:
        if itr > 300:
            print('Warning: Shape failed to project.  Geodesics will be incorrect.')
            return  qnew 

        # Compute Jacobian
        for i in range(n):
            for j in range(n):
                J[i, j] = 3 * np.trapz(qnew[i, :] * qnew[j, :], D)
        J += np.eye(n)

        qnorm = np.zeros(T)
        for i in range(T):
            qnorm[i] = np.linalg.norm(qnew[:, i], ord=2)
        
        # Compute residue
        G = np.zeros(n)
        for i in range(n):
            G[i] = np.trapz((qnew[i, :] * qnorm), D)
        res = -G

        if np.linalg.norm(res, ord=2) < epsilon:
            return qnew 

        cond_J = np.linalg.cond(J)

        if np.isnan(cond_J) or np.isinf(cond_J) or (cond_J < 0.1):
            print('\nProjection may not be accurate\n')
            q = q/(np.sqrt(induced_norm_L2(q)) + np.spacing(1))
            return qnew 
        else:
            x = np.linalg.solve(J, res.T)
            delG = form_basis_normal_a(qnew)
            temp = 0
            for i in range(n):
                temp = temp + x[i] * delG[i] * dt
            qnew += temp
            itr += 1  #Iterator for while loop

    return qnew




