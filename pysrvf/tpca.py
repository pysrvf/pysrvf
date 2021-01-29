import numpy as np
from scipy import linalg
from scipy import stats
import pysrvf.geodesic_utils
from pysrvf.geodesic_utils import project_tangent
import pysrvf.generic_utils
from pysrvf.form_basis_utils import *
from pysrvf.main_mean import get_data_mean

def tpca_from_data(X):
    qmean, pmean, pmean_scaled, Xdata, qarr, alpha_t_arr = get_data_mean(X)

    return


def tpca_from_mean(qmean, tangent_vectors):

    epsilon = 0.0001
    d,n,T = tangent_vectors.shape

    B = form_basis_L2_R3(d,T)
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
    Xproj = project_to_basis(tangent_vectors,G)
    C = np.cov(Xproj[0].T)
    [U, S, V] = linalg.svd(C)

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

    return ret_dict



# covdata.qmean = qmean;
# covdata.alpha_t_array = alpha_t_array;
# covdata.Y = Y;
# covdata.Xproj = Xproj;
# covdata.U = U;
# covdata.S = S;
# covdata.V = V;
# covdata.C = C;
# covdata.Cn = Cn;

X = np.load('../data/3d/bundle_3d_2_1.npy')
qmean, pmean, pmean_scaled, Xdata, qarr, alpha_t_arr = get_data_mean(X)
info = tpca_from_mean(qmean, alpha_t_arr[0])
#print(info['U'])

