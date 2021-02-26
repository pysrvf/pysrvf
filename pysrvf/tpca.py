import numpy as np
from scipy import linalg
from scipy import stats
import pysrvf.geodesic_utils
from pysrvf.geodesic_utils import project_tangent
import pysrvf.generic_utils
from pysrvf.form_basis_utils import *
from pysrvf.main_mean import get_data_mean
from scipy.io import loadmat


def tpca_from_data(X, num_iter=15):
    qmean, pmean, pmean_scaled, Xdata, qarr, alpha_t_arr, gamma_arr = get_data_mean(X, num_iter=num_iter)
    covdata = tpca_from_mean(qmean, alpha_t_arr)
    covdata['gamma_array'] = gamma_arr
    return covdata

def pcacov(C):
    '''
    returns eigenvals, eigenvects, explained variance
    '''
    variance_explained = []
    eigen_values, eigen_vectors = np.linalg.eigh(C)
    for i in eigen_values:
        variance_explained.append((i/sum(eigen_values))*100)
    return eigen_vectors, eigen_values, np.array(variance_explained)


def tpca_from_mean(qmean, tangent_vectors):

    epsilon = 0.0001
    _,n,T = tangent_vectors.shape
    d = 20

    if (n == 2):
        B = form_basis_L2_R2(d,T)
    if (n == 3):
        B = form_basis_L2_R3(d, T)

    B = gram_schmidt(B)
    Bnew = form_basis_of_tangent_space_of_S_at_q(B, qmean)

    G_O_q = form_basis_O_q(B,qmean)
    G_O_q = gram_schmidt(G_O_q)

    # T_q(C) = T_q(S) + T_q(O_q)^{\perp}
    # Subtract the projection of basis of T_q(C) onto T_q(O_q) from itself
    # i.e. basis(T_q(C)) - <basis(T_q(C)), basis(T_q(O_q))> * basis(T_q(O_q))

    Bnew = gram_schmidt(Bnew)
    G = form_basis_of_tangent_space_of_S_at_q(Bnew, G_O_q)
    G = gram_schmidt(G) # Orthogonalize the basis of T_mu(S)

    # From this point onwards G is the Fourier basis for T_mu(S)
    # Project the tangent vectors on this basis
    # -----------

    Y = gram_schmidt(tangent_vectors)

    # project_to_tangent_C_q
    Xproj, X = project_to_basis(tangent_vectors,G)
    C = np.cov(Xproj.T)
    U, S, V = linalg.svd(C)

    sDiag = np.diag(S)
    tmp = np.identity(len(S))
    tmp = epsilon*tmp
    Cn = U*(tmp+sDiag)*U.T

    ret_dict = {}
    ret_dict['Cn'] = Cn
    ret_dict['qmean'] = qmean
    ret_dict['alpha_t_array'] = tangent_vectors
    ret_dict['Y'] = Y
    ret_dict['Xproj'] = Xproj
    ret_dict['U'] = U
    ret_dict['S'] = S
    ret_dict['V'] = V
    ret_dict['C'] = C
    ret_dict['X'] = X
    ret_dict['G'] = G
    ret_dict['Eigproj'] = np.dot(Xproj, U)
    #ret_dict['PC'] = PC
    #ret_dict['Latent'] = Latent
    #ret_dict['Explained'] = Explained

    return ret_dict
