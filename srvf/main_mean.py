import numpy as np 
from srvf_utils import batch_curve_to_q
from find_mean_shape import get_mean

# Load dataset. Assumes format is (N x n x T)
data_dir = '/home/elvis/Documents/BMAP/pysrvf/srvf/Data/2d/dog_curves.npy'
Xdata = batch_curve_to_q(np.load(data_dir))

print(np.linalg.norm(get_mean(Xdata), 2))