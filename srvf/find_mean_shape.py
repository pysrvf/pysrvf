import numpy as np 
from generic_utils import *
from geodesic_utils import *
from form_basis_utils import *
from compute_geodesic import *

def get_mean(qarr):
	'''
	Gets the elastic mean of the collection of SRVF curves
	Inputs:
	- qarr: An (N x n x T) array of SRVF curves
	Outputs:
	- qmean: An (n x T) matrix representing the mean of the curves
	- alpha_arr:
	- alpha_t_arr:
	- norm_alpha_t_mean: 
	- gamma_array:
	- sum_sq_dist_iter:
	- Egeo_array:
	- geo_dist_array:
	'''

	# Constants
	N, n, T = np.shape(qarr)
	stp = 7
	dt = 0.1
	d = 10 # Number of Fourier coefficients divided by 2

	# Initialize mean to extrinsic average
	qmean = projectC(np.mean(qarr, axis = 0))
	qshapes = np.zeros((2, n, T))

	num_itr = 5
	norm_alpha_t_mean = np.zeros(num_itr)
	sum_sq_dist_itr = np.zeros((num_itr, n, T))

	for itr in range(num_itr):
		print('\nIteration {}'.format(itr+1))
		alpha_t_mean = np.zeros((n, T))
		sum_sq_dist = 0
		qshapes[0] = qmean
		for i in range(N):
			qshapes[1] = qarr[i]
			_, alpha_t_arr_i, _, _, _, geo_dist_arr_i = geodesic_distance_all(qshapes, 'all')
			alpha_t_mean += alpha_t_arr_i[0][1] 
			sum_sq_dist += np.square(geo_dist_arr_i)

		alpha_t_mean /= N
		norm_alpha_t_mean[itr] = induced_norm_L2(alpha_t_mean)
		sum_sq_dist_itr[itr] = sum_sq_dist
		qmean, _ = geodesic_flow(qmean, alpha_t_mean, stp)
	
	# Compute geodesics between the mean shape and each of the training shapes
	print('\nGeodesics between mean shape and training shapes...')
	qshapes[0] = qmean 
	alpha_t_arr = np.zeros((stp, N, n, T))
	alpha_arr = np.zeros((stp, N, n, T))
	gamma_arr = np.zeros((N, T-1))
	A_norm_arr = []#np.zeros((N, n))
	E_geo_arr = []#np.zeros((N, n))
	geo_dist_arr = np.zeros((N, n, T))

	for i in range(N):
		qshapes[1] = qarr[i]
		alpha_arr_i, alpha_t_arr_i, A_norm_iter_arr_i, E_geo_C_arr_i, gamma_arr_i, geo_dist_arr_i = \
			geodesic_distance_all(qshapes, 'all')
		alpha_t_arr[i] = alpha_t_arr_i[0]
		alpha_arr[i] = alpha_arr_i[0]
		gamma_arr[i] = np.array(gamma_arr_i)
		A_norm_arr.append(A_norm_iter_arr_i)
		E_geo_arr.append(E_geo_C_arr_i[0])
		geo_dist_arr[i] = geo_dist_arr_i

	return qmean, alpha_arr, alpha_t_arr, norm_alpha_t_mean, gamma_arr, sum_sq_dist_itr, \
		E_geo_arr, geo_dist_arr
