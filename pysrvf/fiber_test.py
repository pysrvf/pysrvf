import numpy as np
from pysrvf.generic_utils import *
from pysrvf.main_mean import *
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
Xdata = np.load('/desktop/BundleRegistrationdata/subject_tracts/subject_1/tract_5.npy')
n, T, N = np.shape(Xdata)
num_samples = 2
perm = np.random.permutation(N)[:num_samples]
Xdata = Xdata[:,:,perm]


# Set subject_first = True if shape of data is (N, n, T)
# Set subject_first = False if shape of data is (n, T, N)
qmean, pmean, pmean_scaled, reformatted_Xdata, _ = get_data_mean(Xdata, subject_first = False) # This changed

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

### Plot data and mean ###
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for num in range(N): # This changed
        x,y,z = reformatted_Xdata[num] # This changed
        ax.scatter(x,y,z, c = 'b')
    #for c in reformatted_Xdata:
    #    ax.plot3D(c[0],c[1],c[2], alpha = 0.4)
        #ax.scatter(pmean_scaled[0,:], pmean_scaled[1,:], pmean_scaled[2,:], 'k--')

plt.show()
