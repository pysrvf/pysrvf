import numpy as np
from generic_utils import *
from geodesic_utils import *
from form_basis_utils import *

def compute_geodesic_C(q0, q1, stp, dt):
	'''
	Computes geodesic from q0 to q1 (after projectC)
	Inputs:
	- q0: An (n x T) matrix representing SRVF curve
	- q1: An (n x T) matrix representing SRVF curve
	- stp: An integer specifying the number of points (minus 1) to sample on geodesic
	- dt: Integer speciying step size
	Outputs:
	- alpha: An ((stp+1) x n x T) array representing the geodesic
	- alpha_t: An ((stp+1) x n x T) array representing the geodesic derivative
	- E: An array of max length 10 where each entry if the palais inner product
	     for each alpha_t iteration 
	- L: The length of the geodesic
	'''
	q1 = regroup(q0, q1)
	q1 = projectC(q1)

	# Initialize path in C by forming a geodesic in Q
	alpha = geodesic_Q(q0, q1, stp)

	itr = 1
	E = [1000]
	v_norm = 100


	while (itr < 10) and (v_norm > 1e-3):
		alpha_t = dAlpha_dt(alpha)
		L = path_length(alpha_t)
		E.append(palais_inner_product(alpha_t, alpha_t))
		w = cov_int_alpha_t(alpha, alpha_t)
		_, t_w_tilde = back_parallel_transport_C(w[stp], alpha)
		v = w - t_w_tilde
		v_norm = palais_inner_product(v, v)
		alpha = path_update(alpha, -v, 0.5)

		alpha[-1] = q1
		itr += 1

	return alpha, alpha_t, E, L