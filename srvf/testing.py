import numpy as np 
import matplotlib.pyplot as plt

curves = np.load('../Data/2d/misc.npy')
N, n, T = np.shape(curves)
new_curves = np.zeros((n, T, N))
for i in range(N):
	new_curves[:,:,i] = curves[i]

np.save('../Data/2d/misc.npy', new_curves)