from pysrvf.tpca import tpca_from_mean
from pysrvf.main_mean import get_data_mean
import numpy as np
from pysrvf.geodesic_utils import geodesic_flow
from pysrvf.generic_utils import q_to_curve, batch_q_to_curve, curve_to_q, batch_curve_to_q
import matplotlib.pyplot as plt

Xdata = np.load('../data/3d/tract_5.npy')
n, T, N = np.shape(Xdata)
num_samples = 20
perm = np.random.permutation(N)[:num_samples]
Xdata = np.transpose(Xdata, (2, 0, 1))
Xdata = Xdata[perm,:,:]

# get_data_mean(Xdata, subject_first = True, num_iter=15)
qmean, pmean, pmean_scaled, Xdata, qarr, alpha_t_arr, gamma_arr = get_data_mean(Xdata, subject_first = True, num_iter=2)

def recon_shape_from_eigen(qmean, alpha_t_arr, covdata, num_eig):
    U = covdata['U']
    Xproj = covdata['Xproj']
    G = covdata['G']
    Xtemp = np.zeros(alpha_t_arr.shape)

    stp = 10
    for ii in range(Xproj.shape[0]):
        Xproj_trunc = np.matmul(Xproj, U[:,:num_eig])
        ghpi = 0
        for jj in range(num_eig):
            gphi = ghpi + Xproj_trunc[ii,jj] * U[:,jj].T

        final_alpha_t = np.zeros((n,T))
        for i in range(G.shape[0]):
            final_alpha_t += gphi[i] * G[i]

        qfinal, alpha_final = geodesic_flow(qmean, final_alpha_t, stp, False)
        Xtemp[ii] = q_to_curve(qfinal)
        

    return Xtemp

#tpca_from_mean(qmean, tangent_vectors)
covdata = tpca_from_mean(qmean, alpha_t_arr)
# Position is not preserved
Xdata_recon_translation_loss = recon_shape_from_eigen(qmean, alpha_t_arr, covdata, 1)

N,_,_ = Xdata.shape
fig = plt.figure()
ax = fig.gca(projection='3d')

Xdata, _ = batch_curve_to_q(Xdata)
Xdata = np.array(Xdata)
Xdata = batch_q_to_curve(Xdata)
Xdata = np.array(Xdata)
print(Xdata.shape)

for i in range(N):
    x,y,z = Xdata[i]
    ax.plot(x,y,z, c='r', alpha = 0.25)


for i in range(N):
    x,y,z = Xdata_recon_translation_loss[i]
    ax.plot(x,y,z, c='b', alpha = 0.25)



plt.show()
