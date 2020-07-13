import numpy as np 
from srvf_utils import *
from karcher_mean import karcher_mean
from resample_curve import resample
from projectC import projectC

def get_mean(curves, to_rescale = True, to_resample = False, resample_size = None, weights = None, D = None):
	'''
	Returns the Karcher mean of a collection of curves
	Inputs:
	- curves: An (N x n x T) array of curves. N curves, represented as (n x T) matrices
	- to_rescale: A boolean denoting whether to normalize the lengths of the curves to 1
	- to_resample: A boolean denoting whether to resample the number of points in the curves. 
				   If true, must give a resample size. 
	- resample_size: If to_resample is true, the new dimension of curves will be
					(N x n x resample_size)
	- weights: An N-dimensional vector with nonnegative entries summing to 1. Denotes the weight
			   for each curve.
	- D: An array of length T denoting the points at which the curves are sampled, i.e., 
		 curves[i]: D -> R^n
	Outputs:
	- mu_p: A (n x T) matrix (or (n x resample_size) if to_resample is true) denoting the Karcher
		    mean of the curves
	- curves: The rescaled/resampled curves
	'''
	if to_rescale:
		curves = rescale_curves(curves)

	if to_resample:
		if resample_size is None:
			print('Error: resample size not given')
			return
		else:
			curves = [resample(c, resample_size) for c in curves]

	N, n, T = np.shape(curves)

	if weights is None:
		weights = np.ones(N)/N 

	# Get SRVF representations
	q_arr = batch_curve_to_q(curves, D)

	# Compute mean
	mu_q = karcher_mean(q_arr, weights, D)

	# Project mean onto space of closed curves
	mu_q = projectC(mu_q, D)

	# Convert back to p-space
	mu_p = q_to_curve(mu_q)

	# Center mean
	mu_p[0,:] = mu_p[0,:] - np.mean(mu_p[0,:])
	mu_p[1,:] = mu_p[1,:] - np.mean(mu_p[1,:])

	return mu_p, curves