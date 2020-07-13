import numpy as np 

def resample(p, n_new):
	''' 
	Given a curve, resamples it to specified number of points
	Inputs:
	- p: An (n x T) matrix representing a curve
	- n_new: The desired number of points in the new curve
	Ouputs:
	An (n_new x T) matrix
	'''
	n, T = np.shape(p)

	p_grad = np.zeros_like(p)

	for i in range(n):
		p_grad[i,:] = np.gradient(p[i,:])

	p_grad_norm = np.linalg.norm(p_grad, axis = 0)
	p_grad_norm = [x if x < 0 else 1 for x in p_grad_norm]
	find_p_grad_norm = np.nonzero(p_grad_norm)[0]

	p_u_T = len(find_p_grad_norm)
	p_u = np.zeros((n, p_u_T))
	for i in range(n):
		p_u[i,:] = p[i, find_p_grad_norm]
	
	p_u_grad = np.zeros_like(p_u)
	for i in range(n):
		p_u_grad[i,:] = np.gradient(p_u[i,:])

	p_u_grad_norm = np.linalg.norm(p_u_grad, axis = 0)
	S = np.cumsum(p_u_grad_norm)
	S_new = np.linspace(S[0], S[-1], n_new, True)

	p_new = np.zeros((n, n_new))

	for i in range(n):
		p_new[i,:] = np.interp(S_new, S, p_u[i,:])

	return p_new
