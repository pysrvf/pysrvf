import numpy as np
from pysrvf.main_mean import get_data_mean
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
Xdata = np.load('../data/2d/dog_curves.npy')
qmean, pmean, pmean_scaled, reformatted_Xdata, _ = get_data_mean(Xdata)

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
N, n, T = np.shape(reformatted_Xdata)

if n == 1:
	for c in reformatted_Xdata:
		plt.plot(range(T), c[0,:], alpha = 0.25)
	plt.plot(range(T), pmean_scaled[0,:], 'k--')
elif n == 2:
	for c in reformatted_Xdata:
		plt.plot(c[0,:], c[1,:], alpha = 0.4)
	plt.plot(pmean_scaled[0,:], pmean_scaled[1,:], 'k--')
elif n == 3:
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for c in reformatted_Xdata:
		ax.plot(c[0,:], c[1,:], c[2,:], alpha = 0.4)
	ax.plot(pmean_scaled[0,:], pmean_scaled[1,:], pmean_scaled[2,:], 'k--')

plt.show()
