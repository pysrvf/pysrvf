import numpy as np 
from projectC import projectC

def get_mean(qarr):
	'''
	Gets the elastic mean of the collection of SRVF curves
	Inputs:
	- qarr: An (N x n x T) array of SRVF curves
	Outputs:
	- qmean: An (n x T) matrix representing the mean of the curves
	- alpha_array:
	- alpha_t_array:
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

	

	return qmean
