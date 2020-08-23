import numpy as np
import warnings
from generic_utils import *
from geodesic_utils import *
from form_basis_utils import *
from scipy.integrate import trapz

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
	q1 = projectC(regroup(q0, q1))

	# Initialize path in C by forming a geodesic in Q
	alpha = geodesic_Q(q0, q1, stp)

	itr = 1
	# E = [1000]
	E = []
	v_norm = 100


	while (itr < 10) and (v_norm > 1e-3):
		alpha_t = dAlpha_dt(alpha, True)
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

def compute_geodesic_B(q1, q2, stp):
	'''
	Computes geodesic from q0 to q1 (after project_B)
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
	theta = np.arccos(inner_product_L2(q1, q2))
	f = q2 - inner_product_L2(q1,q2)*q1
	f = theta*f/induced_norm_L2(f)

	alpha = [project_B(geodesic_sphere(q1, f, tau/stp)) for tau in range(stp+1)]
	alpha_t = dAlpha_dt(alpha, False)
	E = palais_inner_product(alpha_t, alpha_t)
	L = path_length(alpha_t)

	return alpha, alpha_t, E, L

def compute_geodesic_C_factor_D_closed(q1, q2, stp, d, dt):
	'''
	Computes geodesic from q0 to q1 and warping function (for closed curves)
	Inputs:
	- q0: An (n x T) matrix representing SRVF curve
	- q1: An (n x T) matrix representing SRVF curve
	- stp: An integer specifying the number of points (minus 1) to sample on geodesic
	- d: The number of Fourier coefficients (divided by 2)
	- dt: Integer speciying step size
	Outputs:
	- alpha: An ((stp+1) x n x T) array representing the geodesic
	- alpha_t: An ((stp+1) x n x T) array representing the geodesic derivative
	- A_norm_iter: An array of max length 45 ...
	- E_geo_C: An array of max length 45 containint palais inner products
	- gamma: A T-dimensional vector representing the warping of q2 to q1
	- geo_dist: The length of the geodesic
	'''
	itr = 0 
	n, T = np.shape(q1)
	V = form_basis_D(d, T)
	s = np.linspace(0, 2*np.pi, T, True)
	epsilon = 0.1
	diffL = 100

	q2n, _ = initialize_gamma_using_DP(q1, q2, True)

	if (induced_norm_L2(q1 - q2n)**2) < epsilon/15:
		A_norm_iter = 0
		E_geo_C = 0
		gamma = s
		geo_dist = 0
		alpha = np.zeros((stp, n, T)) 
		alpha = [q1 for i in range(stp)]
		alpha_t = np.zeros((stp+1, n, T))
		return alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist

	E_geo_C = []
	A_norm_iter = []
	L_iter = []
	while (itr < 44) and (diffL > epsilon/50):
		q2n = regroup(q1, q2n)
		q2n = projectC(q2n)
		alpha, alpha_t, E, L = compute_geodesic_C(q1, q2n, stp, dt)
		L_iter.append(L)
		E_geo_C.append(E[-1])
		u = alpha_t[stp]
		D_q = form_basis_D_q(V, q2n)
		u_proj, a = project_tgt_D_q(u, D_q) # Need to check this
		g = np.matmul(a, V)
		A_norm_iter.append(trapz(np.square(g), s))

		# Form gamma
		gamma_n = s - epsilon*g 
		gamma_n = gamma_n - gamma_n[0]
		gamma_n = 2*np.pi*gamma_n/np.max(gamma_n)

		if np.sum(gamma_n < 0) or np.sum(np.diff(gamma_n) < 0):
			warnings.warn('Gamma is invalid')
			break 

		q2n = group_action_by_gamma(q2n, gamma_n, True)
		q2n = projectC(q2n)

		itr += 1
		if itr > 1:
			diffL = np.linalg.norm(L_iter[itr-1] - L_iter[itr-2])

	gamma = estimate_gamma(q2n, True)
	geo_dist = L

	return alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist

def compute_geodesic_C_factor_D_open(q1, q2, stp, d, dt):
	'''
	Computes geodesic from q0 to q1 and warping function (for open curves)
	Inputs:
	- q0: An (n x T) matrix representing SRVF curve
	- q1: An (n x T) matrix representing SRVF curve
	- stp: An integer specifying the number of points (minus 1) to sample on geodesic
	- d: The number of Fourier coefficients (divided by 2)
	- dt: Integer speciying step size
	Outputs:
	- alpha: An ((stp+1) x n x T) array representing the geodesic
	- alpha_t: An ((stp+1) x n x T) array representing the geodesic derivative
	- A_norm_iter: An array of max length 45 ...
	- E_geo_C: An array of max length 45 containint palais inner products
	- gamma: A T-dimensional vector representing the warping of q2 to q1
	- geo_dist: The length of the geodesic
	'''
	itr = 0 
	n, T = np.shape(q1)
	V = form_basis_D(d, T)
	s = np.linspace(0, 2*np.pi, T, True)
	epsilon = 1e-6
	diffL = 100

	q1 = project_B(q1)
	q2n = project_B(q2)
	q2n, _ = initialize_gamma_using_DP(q1, q2, False)
	q2n = project_B(q2n)
	alpha, alpha_T, E, L = compute_geodesic_B(q1, q2n, stp)


	if (induced_norm_L2(q1 - q2n)**2) < epsilon/15:
		A_norm_iter = 0
		E_geo_C = 0
		gamma = s
		geo_dist = 0
		alpha = np.zeros((stp, n, T))
		alpha = [q1 for i in range(stp)]
		alpha_t = np.zeros((stp+1, n, T))
		return alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist


	E_geo_C = []
	A_norm_iter = []
	L_iter = []
	while (itr < 44) and (diffL > epsilon/100):
		q2n = project_B(q2n)
		alpha, alpha_t, E, L = compute_geodesic_B(q1, q2n, stp)
		L_iter.append(L)
		E_geo_C.append(E)
		u = alpha_t[stp]
		D_q = form_basis_D_q(V, q2n)
		u_proj, a = project_tgt_D_q(u, D_q) 
		g = np.matmul(a, V)
		A_norm_iter.append(trapz(np.square(g), s))

		# Form gamma
		gamma_n = s - epsilon*g 
		gamma_n = gamma_n - gamma_n[0]
		# gamma_n = 2*np.pi*gamma_n/np.max(gamma_n)

		if np.sum(gamma_n < 0) or np.sum(np.diff(gamma_n) < 0):
			warnings.warn('Gamma is invalid')
			break 

		q2n = group_action_by_gamma(q2n, gamma_n, False)
		q2n = project_B(q2n)

		itr += 1
		if itr > 1:
			diffL = np.linalg.norm(L_iter[itr-1] - L_iter[itr-2])
	gamma = estimate_gamma(q2n, False)
	geo_dist = L

	return alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist

def compute_geodesic_C_factor_D_symm(q1, q2, stp, d, dt):
	'''
	# TODO: Check this once DP_matchSymm becomes available
	Computes geodesic from q0 to q1 and warping function (for open curves)
	Inputs:
	- q0: An (n x T) matrix representing SRVF curve
	- q1: An (n x T) matrix representing SRVF curve
	- stp: An integer specifying the number of points (minus 1) to sample on geodesic
	- d: The number of Fourier coefficients (divided by 2)
	- dt: Integer speciying step size
	Outputs:
	- alpha: An ((stp+1) x n x T) array representing the geodesic
	- alpha_t: An ((stp+1) x n x T) array representing the geodesic derivative
	- A_norm_iter: An array of max length 45 ...
	- E_geo_C: An array of max length 45 containint palais inner products
	- gamma: A T-dimensional vector representing the warping of q2 to q1
	- geo_dist: The length of the geodesic
	'''
	itr = 0 
	n, T = np.shape(q1)
	V = form_basis_D(d, T)
	s = np.linspace(0, 2*np.pi, T, True)
	epsilon = 1e-6
	diffL = 100

	q1 = project_B(q1)
	q2n = project_B(q2)
	q2n, _ = initialize_gamma_using_DP_symm(q1, q2, False)
	q2n = project_B(q2n)
	alpha, alpha_T, E, L = compute_geodesic_B(q1, q2n, stp)


	if (induced_norm_L2(q1 - q2n)**2) < epsilon/15:
		A_norm_iter = 0
		E_geo_C = 0
		gamma = s
		geo_dist = 0
		alpha = np.zeros((stp, n, T))
		alpha = [q1 for i in range(stp)]
		alpha_t = np.zeros((stp+1, n, T))
		return alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist


	E_geo_C = []
	A_norm_iter = []
	L_iter = []
	while (itr < 44) and (diffL > epsilon/100):
		q2n = project_B(q2n)
		alpha, alpha_t, E, L = compute_geodesic_B(q1, q2n, stp)
		L_iter.append(L)
		E_geo_C.append(E)
		u = alpha_t[stp]
		D_q = form_basis_D_q(V, q2n)
		u_proj, a = project_tgt_D_q(u, D_q) 
		g = np.matmul(a, V)
		A_norm_iter.append(trapz(np.square(g), s))

		# Form gamma
		gamma_n = s - epsilon*g 
		gamma_n = gamma_n - gamma_n[0]
		# gamma_n = 2*np.pi*gamma_n/np.max(gamma_n)

		if np.sum(gamma_n < 0) or np.sum(np.diff(gamma_n) < 0):
			warnings.warn('Gamma is invalid')
			break 

		q2n = group_action_by_gamma(q2n, gamma_n, False)
		q2n = project_B(q2n)

		itr += 1
		if itr > 1:
			diffL = np.linalg.norm(L_iter[itr-1] - L_iter[itr-2])
	gamma = estimate_gamma(q2n, False)
	geo_dist = L

	return alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist

def compute_elastic_geodesic(q1, q2, stp, d, dt, data_reg_flag, sym_flag, is_closed):
	'''
	Computes elastic geodesic from q1 to q2 
	Inputs:
	- q0: An (n x T) matrix representing SRVF curve
	- q1: An (n x T) matrix representing SRVF curve
	- stp: An integer specifying the number of points (minus 1) to sample on geodesic
	- d: The number of Fourier coefficients (divided by 2)
	- dt: Integer speciying step size
	- data_reg_flag: 'reg' if shapes are registered, 'nonreg' otherwise
	- sym_flag: 'sym' if to use symmetric
	- is_closed: A boolean indicating whether the original curves are closed
	Outputs:
	- alpha: An ((stp+1) x n x T) array representing the geodesic
	- alpha_t: An ((stp+1) x n x T) array representing the geodesic derivative
	- A_norm_iter: An array of max length 45 ...
	- E_geo_C: An array of max length 45 containint palais inner products
	- gamma: A T-dimensional vector representing the warping of q2 to q1
	- geo_dist: The length of the geodesic
	'''
	n, T = np.shape(q1)

	if is_closed:
		q2 = projectC(q2)
		if data_reg_flag == 'reg':
			q2 = regroup(q1, q2)
			q2, _ = find_best_rotation(q1, q2)
			alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
				compute_geodesic_C_factor_D_closed(q1, q2, stp, d, dt)
		elif data_reg_flag == 'nonreg':
			q2 = regroup(q1, q2)
			q2new = find_rotation_and_seed_unique(q1, q2)
			alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
				compute_geodesic_C_factor_D_closed(q1, q2new, stp, d, dt)
	else:
		q2 = project_B(q2)
		if data_reg_flag == 'reg':
			q2, _ = find_best_rotation(q1, q2)

		if sym_flag == 'sym':
			alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
				compute_geodesic_C_factor_D_symm(q1, q2, stp, d, dt)
		else:
			alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
				compute_geodesic_C_factor_D_open(q1, q2, stp, d, dt)


	return alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist

def geodesic_distance_all(qarr, computation_type, is_closed):
	'''
	'''
	stp = 6
	dt = 0.1
	d = 5

	num_shapes, _, _ = np.shape(qarr)

	alpha_arr = []
	alpha_t_arr = []
	A_norm_iter_arr = []
	E_geo_C_arr = []
	gamma_arr = []
	geo_dist_arr = []

	if computation_type == 'all':
		for i in range(num_shapes):
			q1 = qarr[i]
			for j in np.arange(i+1, num_shapes):
				q2 = qarr[j]
				if is_closed:
					alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
						compute_elastic_geodesic(q1, q2, stp, d, dt, 'nonreg', '', is_closed)
				else:
					alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
						compute_elastic_geodesic(q1, q2, stp, d, dt, 'reg', 'symt', is_closed)
				alpha_arr.append(alpha)
				alpha_t_arr.append(alpha_t)
				A_norm_iter_arr.append(A_norm_iter)
				E_geo_C_arr.append(E_geo_C)
				gamma_arr.append(gamma)
				geo_dist_arr.append(geo_dist)
				# print('{}--{}, {}'.format(i+1, j+1, geo_dist))
	elif computation_type == 'pairwise':
		# Check if array has even number of elements
		if num_shapes%2 == 0:
			for i in range(0, num_shapes, 2):
				q1 = qarr[i]
				q2 = qarr[i+1]
				if is_closed:
					alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
						compute_elastic_geodesic(q1, q2, stp, d, dt, 'nonreg', '', is_closed)
				else:
					alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = \
						compute_elastic_geodesic(q1, q2, stp, d, dt, 'reg', 'sym', is_closed)
				alpha_arr.append(alpha)
				alpha_t_arr.append(alpha_t)
				A_norm_iter_arr.append(A_norm_iter)
				E_geo_C_arr.append(E_geo_C)
				gamma_arr.append(gamma)
				geo_dist_arr.append(geo_dist)
				print('{}--{}, {}'.format(i+1, i+2, geo_dist))
		else:
			print('For the selected option: {},'.format(computation_type) + \
				' qarr should contain an even number of elements.')
	
	return alpha_arr, alpha_t_arr, A_norm_iter_arr, E_geo_C_arr, gamma_arr, geo_dist_arr
