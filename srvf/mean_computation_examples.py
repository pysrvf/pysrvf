import numpy as np 
from main_mean import get_data_mean

### ----- 1d examples ----- ###
# Bumps
# Shape is (N x T)
Xdata = np.load('../Data/1d/two_bumps.npy')
Xdata = Xdata[:4]
qmean, pmean, pmean_scaled, reformatted_Xdata = get_data_mean(Xdata)

### ----- 2d examples ----- ###
# Dog curves
# Shape is (N x n x T)
# Xdata = np.load('../Data/2d/dog_curves.npy')
# qmean, pmean, pmean_scaled, reformatted_Xdata = get_data_mean(Xdata)

# Tract data
# Shape is (n x T x N)
# Xdata = np.load('../Data/1d/hc_FA_data.npy')
# qmean, pmean, reformatted_Xdata = get_data_mean(Xdata, subject_first = False)

### Plot data and mean ###
import matplotlib.pyplot as plt 
N, n, T = np.shape(reformatted_Xdata)

if n == 1:
	for i in range(N):
		plt.plot(range(T), reformatted_Xdata[i][0,:], alpha = 0.25)
	plt.plot(range(T), pmean_scaled[0,:], 'k--')
elif n == 2:
	for i in range(N):
		plt.plot(reformatted_Xdata[i][0,:], reformatted_Xdata[i][1,:], alpha = 0.4)
	# Scale mean to match the scale of original data
	plt.plot(pmean_scaled[0,:], pmean_scaled[1,:], 'k--')
plt.show()