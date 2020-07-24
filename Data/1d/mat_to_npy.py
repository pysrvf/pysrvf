import numpy as np 
from scipy.io import loadmat 

mat_file = loadmat('hc_template')
dat = mat_file['FA_mat']

# Change to (subject x (subjet data))
n, T, N = np.shape(dat)
reform_dat = np.zeros((N, n, T))
for i in range(N):
	reform_dat[i] = dat[:,:,i]

np.save('hc_FA_data.npy', reform_dat)

# norm_sum = 0
# for i in range(N):
# 	norm_sum += np.linalg.norm(reform_dat[i][5,:], 2)

# print(norm_sum)
