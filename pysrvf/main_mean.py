# import numpy as np
# from pysrvf.generic_utils import *
# from pysrvf.find_mean_shape import get_mean

import numpy as np
from pysrvf.generic_utils import *
# from generic_utils import *
from pysrvf.find_mean_shape import get_mean
# from find_mean_shape import get_mean
from scipy.signal import savgol_filter

def get_data_mean(Xdata, subject_first = True):
    '''
    Given a collection of shapes, will return their Karcher mean
    Inputs:
    - Xdata: The collection of shapes. Can be of the following size
                     (N x T): Represents a dataset consisting of N curves/functions each
                                      with T points
                     (N x n x T): Represents a dataset with N shapes, each of which is
                                              of dimension n with T points. In this case, subject_first
                                              should be set to True
                     (n x T x N): Represents a dataset with N shapes, each of which is of
                                              of dimension n with T points. In this case, subejct_first
                                              should be set to False
    - subject_first: A boolean used when Xdata has 3 dimensions. True if shape is
                                     (N x n x T), False if shape is (n x T x N)
    '''
    # Reshape Xdata to be (N x n x T)
    Xdata_shape = np.shape(Xdata)
    if len(Xdata_shape) == 2:
        N, T = Xdata_shape
        Xdata = [np.reshape(c, (1, T)) for c in Xdata]
    elif len(Xdata_shape) == 3:
        if not subject_first:
            n, T, N = np.shape(Xdata)
            Xdata = [Xdata[:,:,i] for i in range(N)]
    else:
        print('Error: Xdata needs to be of size (N x T), (N x n x T), or (n x T x N).')
        exit()

    # If 2d or higher, center curves to origin. The mean is agnostic to translation;
    # this is useful for plotting the shapes
    N, n, T = np.shape(Xdata)
    if n >= 2:
        for i in range(N):
            for j in range(n):
                Xdata[i][j,:] -= np.mean(Xdata[i][j,:])

    # Get SRVF representation
    qarr, is_closed = batch_curve_to_q(Xdata)

    # Number of iterations to run algorithm for
    num_iter = 15

    # Get qmean
    [qmean, alpha_arr, alpha_t_arr, norm_alpha_t_mean, gamma_arr, \
     sum_sq_dist_itr, E_geo_arr, geo_dist_array] = get_mean(qarr, is_closed, num_iter)

    # Convert to native space
    pmean = q_to_curve(qmean)

    # Scale mean to match the scale of original data
    if n >= 2:
        scaling_factor = np.mean([induced_norm_L2(c) for c in Xdata])
        pmean_scaled = scaling_factor*pmean*induced_norm_L2(pmean)
        for i in range(n):
            pmean_scaled[i,:] -= np.mean(pmean_scaled[i,:])
    else:
        pmean_scaled = pmean

    return qmean, pmean, pmean_scaled, Xdata, qarr, alpha_t_arr


def get_deformation_field_from_tangent_vectors(alpha_t_arr):

    K = len(alpha_t_arr)
    _, n, T = np.shape(alpha_t_arr[0])
    alpha_t_mean = np.zeros((n,T))
    for ii in range(K):
        temp = alpha_t_arr[ii]
        alpha_t_mean += temp[1]
    alpha_t_mean /= K
    alpha_t_mag = np.sum(alpha_t_mean**2,axis=0)**(1./2)
    alpha_t_mag = savgol_filter(alpha_t_mag, 9, 2)
    return alpha_t_mag
