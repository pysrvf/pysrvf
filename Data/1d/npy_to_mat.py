import numpy as np 
from scipy.io import savemat 
import matplotlib.pyplot as plt 

npy_dat = np.load('two_bumps.npy')
n, T = np.shape(npy_dat)

# Data is (n x 2T). transform to (2n x T)
# T = int(T/2)
# npy_dat = np.vstack([npy_dat[:,:T], npy_dat[:,T:]])
# np.save('two_bumps.npy', npy_dat)

npy_dat_explode = [np.reshape(c, (1, T)) for c in npy_dat]
savemat('two_bumps.mat', {'bumps': npy_dat_explode})