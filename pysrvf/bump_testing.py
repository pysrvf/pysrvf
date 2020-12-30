import numpy as np 
from scipy.io import loadmat 
from pysrvf.generic_utils import curve_to_q
from pysrvf.generic_utils import q_to_curve
from pysrvf.generic_utils import reparameterize_curve_gamma
from pysrvf.generic_utils import group_action_by_gamma
from dpmatchsrvf import dpmatch
import matplotlib.pyplot as plt

bumps = loadmat('synthesized_bumps.mat')['bumps']

p1 = np.expand_dims(bumps[9,:], axis = 0)
p2 = np.expand_dims(bumps[4,:], axis = 0)
q1 = curve_to_q(p1)
q2 = curve_to_q(p2)

gamma = dpmatch().match(q1, q2)
gamma = 2*np.pi*gamma/np.max(gamma)

# Load matlab gamma file
# matlab_gamma = np.squeeze(loadmat('bump_gamma.mat')['gamma'])
# p2n_matlab = reparameterize_curve_gamma(p2, matlab_gamma)

p2n_pysrvf = reparameterize_curve_gamma(p2, gamma)

plt.plot(p1[0], 'b')
plt.plot(p2[0], 'k')
# plt.plot(p2n_matlab[0], 'g')
plt.plot(p2n_pysrvf[0], 'r')
plt.show()
