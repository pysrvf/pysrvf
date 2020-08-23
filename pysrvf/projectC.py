import numpy as np
import warnings
from pysrvf.generic_utils import *
from pysrvf.form_basis_utils import form_basis_normal_A

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
