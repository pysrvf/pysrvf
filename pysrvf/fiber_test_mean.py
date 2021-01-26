import numpy as np
from pysrvf.main_mean import get_data_mean
from pysrvf.generic_utils import *
import time
# from scipy.io import savemat

### ----- 1d examples ----- ###
# Bumps
# Shape is (N x T)
# Xdata = np.load('../data/1d/two_bumps.npy')
# qmean, pmean, pmean_scaled, reformatted_Xdatam _ = get_data_mean(Xdata)

### ----- 2d examples ----- ###
# Dog curves
# Shape is (N x n x T)
Xdata = np.load('../data/3d/tract_5.npy')
n, T, N = np.shape(Xdata)
num_samples = 100
perm = np.random.permutation(N)[:num_samples]
Xdata = Xdata[:,:,perm]


# Set subject_first = True if shape of data is (N, n, T)
# Set subject_first = False if shape of data is (n, T, N)

qmean, pmean, pmean_scaled, reformatted_Xdata, qarr, alpha_t_arr = get_data_mean(Xdata, subject_first = False) # This changed

qarr_array  = np.array(qarr)
alpha_t_arr_array = np.array(alpha_t_arr)
reformatted_Xdata_array = np.array(reformatted_Xdata)

# 2d parametric curves
# Shape is (n x T x N)
# Xdata = np.load('../data/2d/misc.npy')
# t = time.time()
# # do stuff
# qmean, pmean, pmean_scaled, reformatted_Xdata, _ = get_data_mean(Xdata, subject_first = False)
# print(time.time() - t)

### ----- 3d examples ----- ###
# Xdata = np.load('../data/3d/sine_curves.npy')
# qmean, pmean, pmean_scaled, reformatted_Xdata, _ = get_data_mean(Xdata)

#print(qarr_array.shape)
#print(reformatted_Xdata_array.shape)
#print(alpha_t_arr_array.shape)
### Plot data and mean ###
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = fig.gca(projection='3d')

N, n, T = np.shape(reformatted_Xdata) # This changed (reformmated data will be of shape (N, n, T))

if n == 1:
    for c in reformatted_Xdata:
        plt.plot(range(T), c[0,:], alpha = 0.25)
    plt.plot(range(T), pmean_scaled[0,:], 'k--')

elif n == 2:
    for c in reformatted_Xdata:
        plt.plot(c[0,:], c[1,:], alpha = 0.4)
    plt.plot(pmean_scaled[0,:], pmean_scaled[1,:], 'k--')

elif n == 3:
    for num in range(N): # This changed
        x,y,z = reformatted_Xdata[num] # This changed
        ax.plot(x,y,z, c = 'b', alpha = 0.25)
    x, y, z = np.mean(reformatted_Xdata, axis = 0)
    ax.plot(x,y,z, c = 'r')
    a, b, c = pmean_scaled
    ax.plot(a,b,c, c = 'k')
    plt.show()

'''
    # srvf plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for num in range(N):
        x,y,z = qarr[num]
        ax.plot(x,y,z, c ='b', alpha = 0.25)
    x,y,z = qmean
    ax.plot(x,y,z, c = 'k')

    a,b,c = np.mean(qarr, axis = 0)
    ax.plot(a,b,c, c = 'r')
    plt.show()
'''

'''
tangent_space = project_to_tangent_C_q(qarr,qmean)
print(np.array(tangent_space).shape)


fig = plt.figure()
ax = fig.gca(projection='3d')
for num in range(N):
    x,y,z = tangent_space[num]
    ax.plot(x,y,z, c ='r', alpha = 0.25)
#x,y,z = qmean
a,b,c = qmean
ax.plot(a,b,c, c = 'k')
plt.show()
'''

