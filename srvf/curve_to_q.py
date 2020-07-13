import numpy as np 

def inner_prod_q(q1, q2, T):
	return np.trapz(np.sum(np.multiply(q1,q2), axis = 0), np.linspace(0, 2*np.pi, T, True))

def project_B(q, T):
	return q/np.sqrt(inner_prod_q(q, q, T))

def curve_to_q(p):
	n, T = np.shape(p)

	beta_dot = np.zeros((n,T))
	q = np.zeros((n,T))

	for i in range(n):
		beta_dot[i,:] = np.gradient(p[i,:], 2*np.pi/T)

	eps = np.finfo(float).eps
	for i in range(T):
		q[:,i] = beta_dot[:,i]/(np.sqrt(np.linalg.norm(beta_dot[:,i])) + eps)

	q = project_B(q, T)

	return q

def batch_curves(curves):
	m, n = np.shape(curves)
	q_space_curves = np.zeros((m,n))

	for i in range(m):
		curve = np.reshape(curves[i,:], (1,n))
		q_space_curves[i,:] = curve_to_q(curve)

	return q_space_curves