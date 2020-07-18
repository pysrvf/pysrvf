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

# alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = compute_elastic_geodesic(u, v, 6, 5, .1, 'nonreg')
start = time.time()
alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = geodesic_distance_all(Xdata, 'all')
end = time.time()
print('Elapsed time is {} seconds'.format(end - start))

# print(alpha)
# print('-----')
# print(alpha_t)
# print('-----')
# print(A_norm_iter)
# print('-----')
# print(E_geo_C)
# print('-----')
# print(gamma)
# print('-----')
# print(geo_dist)
