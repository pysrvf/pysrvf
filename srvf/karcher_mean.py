import numpy as np 
from srvf_utils import *

def obj_func_grad(x, q_arr, weights, p, D = None):
	'''
	Computes the gradient of the objective function defining the Karcher mean
	Inputs:
	- x: An (n x T) matrix representation of the current gradient descent
		 minimizer estimate
	- q_arr: A (N x n x T) array containing all SRVF curves
	- weights: An (N x 1) weight vector 
	- p: A scalar denoting the objective function parameter p
	- D: A vector denoting the domain of the curves, i.e., q_i: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	A (n x T) matrix representing the gradient evaluated at x
	'''
	grad_x = np.zeros_like(x)
	for i,x_i in enumerate(q_arr):
		w_i = weights[i]
		d_p = distance_function(x, x_i)**(p-2)
		inv_exp = inverse_exponential_map(x, x_i, D)
		grad_x -= w_i*d_p*inv_exp
	
	return grad_x

def gradient_descent(q_arr, weights, p, tol, max_iter, D = None):
	'''
	Performs gradient descent to estimate the objective function's minimizer
	Inputs:
	- x: An (n x T) matrix representation of the current gradient descent
		 minimizer estimate
	- q_arr: A (N x n x T) array containing all SRVF curves
	- weights: An (N x 1) weight vector 
	- p: A scalar denoting the objective function parameter p
	- tol: A scalar in (0, 1) denoting the termination tolerance
	- D: A vector denoting the domain of the curves, i.e., q_i: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	An (n x T) matrix representing the gradient descent estimate of the minimizer
	'''

	# Initialize to first element in q_arr
	xk = q_arr[0]

	# Step size
	t_k = 1#0.01

	# Initial value of gradient
	grad_x0 = obj_func_grad(xk, q_arr, weights, p, D)
	grad_xk = obj_func_grad(xk, q_arr, weights, p, D)
	converged = np.linalg.norm(grad_xk) <= tol
	k = 0 

	eps = 1e-8
	eps_mat = np.ones_like(grad_x0)*eps
	grad_x0_divisor = grad_x0 + eps_mat
	
	while not converged:
		xk = exponential_map(xk, -t_k*grad_xk)
		grad_xk = obj_func_grad(xk, q_arr, weights, p, D)
		k += 1
		# converged = (np.linalg.norm(np.divide(grad_xk, grad_x0_divisor)) <= tol) or (k > max_iter)
		converged = (np.linalg.norm(grad_xk) <= tol) or (k >= max_iter)
	
	if k >= max_iter:
		print('Warning: Karcher mean failed to converge in {} steps'.format(int(max_iter)))

	return xk 

def karcher_mean(q_arr, weights, D = None):
	''' Computes the Karcher mean of the given array of SRVF curves
	Inputs:
	- q_arr: A (N x n x T) array containing all SRVF curves
	- weights: An (N x 1) weight vector 
	- D: A vector denoting the domain of the curves, i.e., q_i: D -> R^n. If none, will
		 take D = [0, 2pi] discretized into T points
	Outputs:
	An (n x T) matrix representing the Karcher mean
	'''
	p = 2
	tol = 1e-8
	max_iter = 5e3

	return gradient_descent(q_arr, weights, p, tol, max_iter, D)