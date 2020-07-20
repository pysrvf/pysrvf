import numpy as np 
from generic_utils import *
from geodesic_utils import *
from form_basis_utils import *
from compute_geodesic import *
import time

w = np.array([[1,2,3,2,1], [-1,0,1,2,3]])
v = np.array([[5,6,7,8,1], [0,-1,5,1,4]])
u = np.array([[0.01, -0.3, 4, 8, -2], [2, 5, 3, -1, 2]])

data_dir = '/home/elvis/Documents/BMAP/pysrvf/Data/2d/dog_curves.npy'
Xdata = batch_curve_to_q(np.load(data_dir))

# start = time.time()
# alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = geodesic_distance_all(Xdata, 'all')
# end = time.time()
# print('Elapsed time is {} seconds'.format(end - start))


# alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = compute_elastic_geodesic(u,v,6,5,0.1,'nonreg')

# alpha_arr, alpha_t_arr, A_norm_iter_arr, E_geo_C_arr, gamma_arr, geo_dist_arr = geodesic_distance_all(Xdata, 'all')

# alpha_arr = np.array(alpha_arr)
# norm_sum = 0
# a, b, c, d = np.shape(alpha_arr)
# for i in range(a):
# 	for j in range(b):
# 		norm_sum += np.linalg.norm(alpha_arr[i,j,:,:], 2)

# print(norm_sum/100)

# alpha: good
# alpha_t: good
alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = compute_geodesic_C_factor_D(Xdata[4], Xdata[6], 6, 10, 0.1)
norm_sum = np.sum([np.linalg.norm(alpha_i, 2) for alpha_i in alpha_t])
print(norm_sum)

# q2n, _ = initialize_gamma_using_DP(Xdata[4], Xdata[6])
# print(np.linalg.norm(q2n, 2))

# import struct
# f = open('gamma.dat', 'rb')
# T = struct.unpack('i', f.read(4))[0]
# gamma = np.array(struct.unpack('f'*T,  f.read()))
# gamma = 2*np.pi*gamma/np.max(gamma)
# print(np.linalg.norm(gamma, 2))
# f.close()
