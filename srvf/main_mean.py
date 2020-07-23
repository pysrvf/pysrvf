import numpy as np 
from generic_utils import *
from find_mean_shape import get_mean
import time


# Load dataset. Assumes format is (N x n x T)
# data_dir = '/home/elvis/Documents/BMAP/pysrvf/Data/2d/dog_curves.npy'
# qarr, is_closed = batch_curve_to_q(np.load(data_dir))

# Tract data
Xdata = np.load('../Data/1d/hc_FA_data.npy')
N, n, T = np.shape(Xdata)
tract_to_use = 15
Xdata = [np.reshape(c, (1, T)) for c in Xdata[:,tract_to_use-1,:]]
qarr, is_closed = batch_curve_to_q(Xdata)

start = time.time()
[qmean, alpha_arr, alpha_t_arr, norm_alpha_t_mean, gamma_arr, sum_sq_dist_itr, E_geo_arr, geo_dist_array] = get_mean(qarr, is_closed)
end = time.time()
print('Elapsed time is {}.'.format(end-start))

import matplotlib.pyplot as plt
plt.plot(range(T), qmean[0,:])
plt.show()