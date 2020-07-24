import numpy as np 
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

def get_3d_sine_curve(phi, T):
	curve = np.zeros((3, T))
	t_lin = np.linspace(0, 2*np.pi, T, True)

	curve[0,:] = np.cos(t_lin)
	curve[1,:] = np.sin(t_lin)
	curve[2,:] = np.cos(phi*t_lin)

	return curve

T = 300
phi_vals = [1, 2, 4, 6, 8]
curves = [get_3d_sine_curve(phi, T) for phi in phi_vals]

# shift = 0
# for c in curves:
# 	ax.plot(c[0,:], c[1,:], c[2,:]+shift)
# 	shift += 1
# plt.show()

print(np.shape(curves))
# np.save('../Data/3d/sine_curves.npy', curves)