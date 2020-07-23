import numpy as np 
from generic_utils import *
from geodesic_utils import *
from form_basis_utils import *
from compute_geodesic import *
import time

w = np.array([[1,2,3,2,1], [-1,0,1,2,3]])
v = np.array([[5,6,7,8,1], [0,-1,5,1,4]])
u = np.array([[0.01, -0.3, 4, 8, -2], [2, 5, 3, -1, 2]])

Xdata = np.load('../Data/1d/hc_FA_data.npy')

# Get fifth tract
N, n, T = np.shape(Xdata)
Xdata = [np.reshape(c, (1, T)) for c in Xdata[:,4,:]]

# To SRVF
qarr = batch_curve_to_q(Xdata)

alpha, alpha_t, A_norm_iter, E_geo_C, gamma, geo_dist = compute_geodesic_C_factor_D_open(qarr[0], qarr[1], 6, 5, 5)
print(geo_dist)
# print(gamma)
# print(gamma)
# print(np.linalg.norm(estimate_gamma(qarr[1], False), 2))
# print(np.linalg.norm(group_action_by_gamma(qarr[0], np.linspace(0, 2*np.pi, 100, True), False), 2))
# print(np.linalg.norm(np.gradient(np.linspace(0, 2*np.pi, 100, True), 2*np.pi/100), 2))