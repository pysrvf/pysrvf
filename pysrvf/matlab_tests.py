# This is to test against the matlab code
import numpy as np
from pysrvf import form_basis_utils as fbu
#from pysrvf.form_basis_utils import *
#form_basis_D, form_basis_D_q, project_tgt_D_q,project_tangent, gram_schmidt, form_basis_L2_R3


# Usually number of curves
d = 3

# Usually number of points in a curve
T = 4

# dimension of space i.e. R^n
n = 3

# inputs d, T positive integers
diffeo_fourier_basis = fbu.form_basis_D(d,T)
# dimension is (2*d) x T

# dimenstion is 1 x 4
t = np.linspace(0, 2*np.pi, T)

# dimension is (n,) i.e. 1 x n
q = np.arange(1,n+1)
#dimension n x T
q = np.outer(q,t)

# inputs Fouier_basis ((2*d) x T), q srvf (n x T)
D_q = fbu.form_basis_D_q(diffeo_fourier_basis,q)
# dimension is ((2*d) x n x T)

# assign q to u
u = q

# input u (n x T), D_q ((2*d) x n x T)
u_proj, a = fbu.project_tgt_D_q(u,D_q)
print(u_proj)
print(a)
# output u_proj (n X T) projection of u, a (1 x 2*d) array of Fouier Coeff

# Set X to D_q for sake of example
X = D_q


# Accidentally skipped over project_tangent(f,q,is_closed)
# -->> Make sure to review this.
# input f (n x T), q (n x T), is_closed = False
proj_tang = fbu.project_tangent(u,q,False)
#print(proj_tang.shape)
#print(proj_tang)

# Figure out what to do with varargin parameters
# Left off 1:31 om 1/22/2021
#print(gram_schmidt(X).shape)
#print(gram_schmidt(X))


# -----------
#Currently the shape is correct, but the values seem off 11:12am (1/23/2021)
#basis_L2_R3 = fbu.form_basis_L2_R3(d,T)

#print(np.zeros((6*d,3,T)).shape)
#print(basis_L2_R3.shape)
#print(basis_L2_R3[0].shape)
#print(basis_L2_R3)

#------------
# need to read Shantanu's papers to understand more about form_basis_of_tangent_space_of_S_at_q
# 11:46am (1/23/2021)


#------------
# project_to_basis doesn't seem to have matlab eqiv in fn_match_mac
# UPDATE: 11:53am (1/23/2021) It's in the tpca_basis folder







