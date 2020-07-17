import numpy as np 

from compute_geodesic import *

w = np.array([[1,2,3,2,1], [-1,0,1,2,3]])
v = np.array([[5,6,7,8,1], [0,-1,5,1,4]])
u = np.array([[0.01, -0.3, 4, 8, -2], [2, 5, 3, -1, 2]])

# V = form_basis_D(5, 5)
# D_q = form_basis_D_q(V, w)
# print(project_tgt_D_q(v, D_q))


print(compute_geodesic_C(v, u, 6, 0.5))
# print(geodesic_Q(v, u, 6))

# alpha = geodesic_Q(v, u, 6)
# alpha_t = dAlpha_dt(alpha)
# L = path_length(alpha_t)
# print(L)