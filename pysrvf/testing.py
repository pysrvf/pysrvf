import numpy as np 
import matplotlib.pyplot as plt

curves = np.load('../Data/2d/misc.npy')
N, n, T = np.shape(curves)
new_curves = np.zeros((n, T, N))
for i in range(N):
	new_curves[:, :, i] = curves[i]

np.save('../data/2d/misc.npy', new_curves)



import os
import numpy as np
from dpmatchsrvf import dpmatch
from scipy import io
import time
import generic_utils
qarray = io.loadmat('../data/q.mat')
qarr = np.array([qarray['q1'], qarray['q2']])

t = time.time()
generic_utils.save_q_shapes('DPshapedata.dat', qarr)
# Call to get gamma
os.system('./DP_Shape_Match_SRVF_nDim DPshapedata.dat gamma.dat')
gamma = generic_utils.load_gamma('gamma.dat')
print(time.time() - t)

t = time.time()
gamma = dpmatch().match(qarr[0], qarr[1])
print(time.time() - t)




