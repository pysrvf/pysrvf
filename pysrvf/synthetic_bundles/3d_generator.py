import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def gen_bundle(amp_min, amp_max, T, curves_in_bundle):
    amplitudes  = np.random.uniform(amp_min, amp_max, curves_in_bundle)
    bundle = np.zeros((curves_in_bundle,T,3))
    for i in range(curves_in_bundle):
        for x in np.linspace(0, np.pi, T):
            bundle[i,:,:] = np.array([amplitudes[i]*(x**2 + np.sin(3*x) + 
            np.sin(5*x)),amplitudes[i]*(x + np.cos(3*x) + np.cos(5*x)), 
            amplitudes[i]*np.sin(x)])

    return bundle

bundle_3d_1 = gen_bundle(0.75,0.6,100,50)
print(bundle_3d_1.shape)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection='3d')

curves_in_bundle, num_samples, dimension  = bundle_3d_1.shape
sample_indices = np.random.permutation(num_samples) # 10
curve_indices  = np.random.permutation(curves_in_bundle) # 5

print(sample_indices.shape)

#for c in curve_indices:
#    print(bundle_3d_1[c,:,:])

#for s in sample_indices:
#    print(bundle_3d_1[:,s,:])

#for w in range(3):
#    print(bundle_3d_1[:,:,w])



#for w in range(3):
#    for c in curve_indices:
#        x,y,z = bundle_3d_1[c,:,:][w]
        #print(bundle_3d_1[c,:,:][w])
        #print(x,y,z)
#        ax.scatter(x,y,z)

#plt.show()
#for idy in sample_other:
#    for idx in sample_indices:
#        x,y,z = bundle_3d_1[:,:,:][idy]
#        ax.plot3D(x,y,z)

#plt.show()







