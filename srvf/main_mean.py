import numpy as np 
from generic_utils import *
from find_mean_shape import get_mean
import time


# Load dataset. Assumes format is (N x n x T)
data_dir = '/home/elvis/Documents/BMAP/pysrvf/Data/2d/dog_curves.npy'
Xdata = batch_curve_to_q(np.load(data_dir))

start = time.time()
[qmean, alpha_arr, alpha_t_arr, norm_alpha_t_mean, gamma_arr, sum_sq_dist_itr, E_geo_arr, geo_dist_array] = get_mean(Xdata)
end = time.time()
print('Elapsed time is {}.'.format(end-start))