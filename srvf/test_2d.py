import numpy as np 
import matplotlib.pyplot as plt
from get_curve_collection_mean import get_mean

# Ellipses
# shape_dir = 'Data/2d/ellipses.npy'

# Dogs (samll)
shape_dir = 'Data/2d/dog_curves.npy'

# Horses (big) (No rotation, so won't give good results)
# shape_dir = 'Data/2d/DogHorse/horses.npy'

# Dogs (big)(No rotation, so won't give good results)
# shape_dir2 = 'Data/2d/DogHorse/dogs.npy'

# Square
# shape_dir = 'Data/2d/square_pair2.npy'

# Misc
# shape_dir = 'Data/2d/misc.npy'

shapes = np.load(shape_dir)

N = 2
K, n, T = np.shape(shapes)
perm_indices = np.random.permutation(K)

resample_size = 300
shape_arr = [shapes[i] for i in perm_indices[:N]]

print('Computing mean...')
mu_p, shape_arr = get_mean(shape_arr, to_resample = True, resample_size = resample_size)

print('Plotting...')
for i in range(N):
	plt.plot(shape_arr[i][0,:], shape_arr[i][1,:])

plt.plot(mu_p[0,:], mu_p[1,:], 'k--')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()