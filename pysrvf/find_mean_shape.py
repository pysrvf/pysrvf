import numpy as np 
from pysrvf.generic_utils import *
from pysrvf.geodesic_utils import *
from pysrvf.form_basis_utils import *
from pysrvf.compute_geodesic import *
from tqdm import trange

def get_mean(qarr, is_closed, num_itr):
  '''
  Gets the elastic mean of the collection of SRVF curves
  Inputs:
  - qarr: An (N x n x T) array of SRVF curves
  - is_closed: A boolean indicating whether original curves were closed
  - num_itr: The number of iterations to run the algorithm for 
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

  stp = 6
  dt = 0.1
  d = 5 # Number of Fourier coefficients divided by 2

  # Initialize mean to extrinsic average
  qmean = np.mean(qarr, axis = 0)

  if is_closed:
    qmean = projectC(qmean)
  else:
    qmean = project_B(qmean)

  qshapes = np.zeros((2, n, T))

  norm_alpha_t_mean = np.zeros(num_itr)
  sum_sq_dist_itr = np.zeros((num_itr, n, T))

  for itr in trange(num_itr, desc = 'Iteration'):
    # print('\nIteration {}'.format(itr+1))
    alpha_t_mean = np.zeros((n, T))
    sum_sq_dist = 0
    qshapes[0] = qmean
    for i in trange(N, desc = 'Shapes', leave = False):#range(N):
      qshapes[1] = qarr[i]
      _, alpha_t_arr_i, _, _, _, geo_dist_arr_i = geodesic_distance_all(qshapes, 'all', is_closed)
      alpha_t_mean += alpha_t_arr_i[0][1]
      sum_sq_dist += np.square(geo_dist_arr_i)
    
    for itr in trange(num_itr, desc = 'Iteration'):
      # print('\nIteration {}'.format(itr+1))
      alpha_t_mean = np.zeros((n, T))
      sum_sq_dist = 0
      qshapes[0] = qmean
    # for i in trange(N, desc = 'Shapes', leave = False):#range(N):
    for i in range(N):  # range(N):
      qshapes[1] = qarr[i]
      _, alpha_t_arr_i, _, _, _, geo_dist_arr_i = geodesic_distance_all(qshapes, 'pairwise', is_closed)
      alpha_t_mean += alpha_t_arr_i[0][1]
      sum_sq_dist += np.square(geo_dist_arr_i)

    alpha_t_mean /= N
    norm_alpha_t_mean[itr] = induced_norm_L2(alpha_t_mean)
    sum_sq_dist_itr[itr] = sum_sq_dist
    qmean, _ = geodesic_flow(qmean, alpha_t_mean, stp, is_closed)


  # Compute geodesics between the mean shape and each of the training shapes
  print('\nComputing geodesics between mean shape and training shapes...')
  qshapes[0] = qmean
  alpha_t_arr = []
  alpha_arr = []
  gamma_arr = []
  A_norm_arr = []
  E_geo_arr = []
  geo_dist_arr = []

  for i in trange(N):
    qshapes[1] = qarr[i]
    alpha_arr_i, alpha_t_arr_i, A_norm_iter_arr_i, E_geo_C_arr_i, gamma_arr_i, geo_dist_arr_i = \
      geodesic_distance_all(qshapes, 'all', is_closed)
    alpha_t_arr.append(alpha_t_arr_i[0])
    alpha_arr.append(alpha_arr_i[0])
    gamma_arr.append(np.array(gamma_arr_i))
    A_norm_arr.append(A_norm_iter_arr_i)
    E_geo_arr.append(E_geo_C_arr_i[0])
    geo_dist_arr.append(geo_dist_arr_i)

  return qmean, alpha_arr, alpha_t_arr, norm_alpha_t_mean, gamma_arr, sum_sq_dist_itr, \
		E_geo_arr, geo_dist_arr
