import numpy as np 
from generic_utils import batch_curve_to_q
from find_mean_shape import get_mean

from generic_utils import load_gamma
from generic_utils import initialize_gamma_using_DP
from generic_utils import group_action_by_gamma

# Load dataset. Assumes format is (N x n x T)
data_dir = '/home/elvis/Documents/BMAP/pysrvf/Data/2d/dog_curves.npy'
Xdata = batch_curve_to_q(np.load(data_dir))

q1 = Xdata[0]
q2 = Xdata[1]

initialize_gamma_using_DP(q1, q2)
gamma = load_gamma('gamma.dat')
print(group_action_by_gamma(q2, gamma))
# print(np.linalg.norm(get_mean(Xdata), 2))