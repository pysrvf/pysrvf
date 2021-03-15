import numpy as np
import matplotlib.pyplot as plt
from pysrvf.main_mean import get_data_mean
from pysrvf.generic_utils import *

def get_3d_bundle(curves_in_bundle = 2, random_state = 1):
  Xdata = np.load('tract_5.npy')
  n, T, N = np.shape(Xdata)
  num_samples = curves_in_bundle
  np.random.RandomState(random_state)
  perm = np.random.permutation(N)[:num_samples]
  Xdata = Xdata[:,:,perm]

  # Reformat Xdata
  qmean, pmean, pmean_scaled, reformatted_Xdata, qarr, alpha_t_arr, gamma_arr = get_data_mean(Xdata, subject_first = False)
  reformatted_Xdata = np.array(reformatted_Xdata)
  N, n, T = np.shape(reformatted_Xdata)

  # Form new_bundle
  new_bundle = np.zeros((N, n, T))
  for i in range(N):
    new_bundle[i:,:] = reformatted_Xdata[i]

  return new_bundle

# new_bundle is a 3d bundle, has n curves, and random state is 1
# new_bundle has shape (N,n,T)


#Saving New Bundle
bundle_3d_10_curves = get_3d_bundle(10,1)
np.save('bundle_3d_10_curves.npy',bundle_3d_10_curves)

#Plotting new_bundle
'''
Xdata = np.load('bundle_3d_2_1.npy')
N,_,_ = Xdata.shape
fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(N):
  x,y,z = Xdata[i]
  ax.plot(x,y,z, c='b', alpha = 0.25)
plt.show()
'''
