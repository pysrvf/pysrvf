import numpy as np 
import warnings
from generic_utils import form_basis_normal_A
from generic_utils import form_basis_D
from generic_utils import shiftF
from generic_utils import regroup
from generic_utils import load_gamma
from generic_utils import initialize_gamma_using_DP
from generic_utils import group_action_by_gamma
from projectC import projectC

w = np.array([[1,2,3,2,1], [-1,0,1,2,3]])
v = np.array([[5,6,7,8,1], [0,-1,5,1,4]])

# initialize_gamma_using_DP(w, v)
gamma = load_gamma('gamma.dat')
print(group_action_by_gamma(w, gamma))
