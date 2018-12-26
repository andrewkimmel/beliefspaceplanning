# Copyright (C) 2015 Philipp Baumgaertel
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE.txt file for details.


import numpy as np
from numpy.linalg import det, cholesky
from scipy.linalg import cho_solve, inv, solve_triangular
#from numpy import dot
import traceback

#Scipy inv seems faster than numpy inv and cho_solve for symmetric matrices
# However cholesky decomposition might be numerically more stable

#TODO!!!: revisit matrix multiplication complexity
class Dot(object):
	"""
	A class to inspect the matrix multiplication complexity
	"""
	_a = 0
	_b = 0
	_c = 0
	_path = ""
	_line = ""
	_in_func = ""

	# def __init__(self):
	# 	self.a = 0
	# 	self.b = 0
	# 	self.c = 0
	# 	self.path = ""
	# 	self.line = ""
	# 	self.in_func = ""

	def __call__(self,A,B):
		"""
		Usage: Like the original np.dot function
		It tracks the matrix multiplication complexity and gives a stacktrace of the most complex matrix multiplication within some code
		:param A: numpy Matrix
		:param B: numpy Matrix
		:return: numpy.dot(A,B)
		"""
		la = len(A.shape)
		lb = len(B.shape)
		n = 1
		o = 1
		m1 = 1
		m2 = 1
		if la == 2:
			n,m1 = A.shape
		else:
			m1 = A.shape[0]

		if lb == 2:
			m2,o = B.shape
		else:
			m2 = B.shape[0]

		if n*m1*o > Dot._a*Dot._b*Dot._c:
			stack = traceback.extract_stack()[-2:]
			Dot._path, Dot._line, Dot._in_func, _instr = stack[0]
			Dot._a = n
			Dot._b = m1
			Dot._c = o


		assert(m1 == m2)
		return np.dot(A,B)

	def reset(self):
		"""
		Reset the gathered statistics
		"""
		Dot._a = 0
		Dot._b = 0
		Dot._c = 0
		Dot._path = ""
		Dot._line = ""
		Dot._in_func = ""
	def __repr__(self):
		return str(Dot._a) + "x" + str(Dot._b) + "x" + str(Dot._c) + ' called from %s in func %s at line %s' % (Dot._path, Dot._in_func, Dot._line)

dot = Dot()

def dldot(a,B):
	"""
	:param a: diagonal of a diagonal matrix
	:param B: Matrix
	"""
	return (a*B.T).T

def drdot(A,b):
	"""
	:param A: Matrix
	:param b: diagonal of a diagonal matrix
	"""
	return A*b

def tracedot(A,B):
	"""

	:param A: Matrix
	:param B: Matrix
	:return: trace(dot(A,B))
	"""
	#assert np.allclose(np.dot(np.ravel(A.T),np.ravel(B)),np.trace(np.dot(A,B)))
	return np.dot(np.ravel(A.T),np.ravel(B))

class Covariance(object):
	"""
	Superclass for all covariance functions
	"""

	def __init__(self):
		pass


	def __call__(self,xi,xj,theta):
		"""
		:param xi: d-dimensional vector
		:param xj: d-dimensional vector
		:param theta: hyperparameters
		:return: covariance between xi and xj
		"""
		pass

	def get_theta(self,x,t):
		"""
		Guesses the initial theta vector for the hyperparameter optimization step

		:return: initial theta vector
		"""
		pass

	def cov_matrix_ij(self,xi,xj,theta):
		"""

		:param xi: list of d-dimensional vectors of size N1
		:param xj: list of d-dimensional vectors of size N2
		:param theta: hyperparameters
		:return: N1xN2 covariance matrix between xi and xj
		"""
		ni = len(xi)
		nj = len(xj)
		K = np.zeros((ni, nj))

		for i in range(ni):
			for j in range(nj):
				K[i, j] = self(xi[i], xj[j], theta)
		return K


	def cov_matrix(self,x,theta):
		"""

		:param x: list of d-dimensional vectors of size N
		:param theta: hyperparameters
		:return: NxN covariance matrix
		"""
		n,dim = np.shape(x)

		return self.cov_matrix_ij(x,x,theta)


	def inv_cov_matrix(self,x,theta,cov_matrix=None):
		"""

		:param x: list of d-dimensional vectors
		:param theta: hyperparameters
		:param cov_matrix: invert this precalculated cov matrix for x and theta
		:return: inverse of the covariance matrix
		"""
		if cov_matrix is None:

			K = np.array(self.cov_matrix(x, theta))
			m=len(K)

			try:
				# return inv(K)
				return np.linalg.pinv(K) # Avishai
			except:# ValueError: # Avishai - Removed the ValueError
				#Inversion done right
				L = cholesky(K+np.eye(m)*1e-5)
				L_inv = solve_triangular(L,np.eye(m),lower=True)
				K_inv = dot(L_inv.T,L_inv)
				return K_inv
		else:
			return inv(cov_matrix)

	def _log_det_cov_matrix(self,x,theta):
		"""
		:param x: list of d-dimensional vectors
		:param theta: hyperparameters
		:return: logarithm of the determinant of the cov matrix
		"""
		return np.linalg.slogdet(self.cov_matrix(x,theta))[1]

	def _negativeloglikelihood(self,x,t,theta):
		"""
		:param x: list of d-dimensional vectors
		:param t: list of responses
		:param theta: hyperparameters
		:return: negative loglikelihood
		"""

		N = len(x)
		logdetK = self._log_det_cov_matrix(x, theta)
		invK = self.inv_cov_matrix(x, theta)

		try:
				#print "t'*inv(K)*t ", dot(t.T, dot(invK, t))
			nll = N / 2.0 * np.log(2 * np.pi) + 0.5 * logdetK + 0.5 * dot(t.T, dot(invK, t))

		except (np.linalg.linalg.LinAlgError, RuntimeWarning, ZeroDivisionError,ValueError):
			nll = 1.0e+20

		return nll


	def _d_cov_d_theta(self,xi,xj,theta,j):
		"""

		:param xi: d-dimensional vector
		:param xj: d-dimensional vector
		:param theta: hyperparameters
		:param j: the part of theta to derive by
		:return: derivative of the covariance d theta_j
		"""
		eps = 1e-5

		d = np.zeros(len(theta))
		d[j] = eps
		return (self(xi,xj,theta+d)-self(xi,xj,theta-d))/(2*eps)


	def _d_cov_matrix_d_theta_ij(self,xi,xj,theta,j):
		"""

		:param xi: list of d-dimensional vectors
		:param xj: list of d-dimensional vectors
		:param theta: hyperparameters
		:param j: the part of theta to derive by
		:return: derivative of the covariance matrix d theta_j
		"""
		ni = len(xi)
		nj = len(xj)

		K = np.zeros((ni, nj))

		for i1 in range(ni):
			for i2 in range(nj):
				K[i1, i2] = self._d_cov_d_theta(xi[i1], xj[i2], theta,j)
		return K


	def _d_cov_matrix_d_theta(self,x,theta,j):
		"""

		:param x: list of d-dimensional vectors
		:param theta: hyperparameters
		:return: derivative of the covariance matrix d theta_j
		"""

		return self._d_cov_matrix_d_theta_ij(x,x,theta,j)


	def _d_nll_d_theta(self,x,t,theta):
		"""

		:param x: list of d-dimensional vectors
		:param t: list of responses
		:param theta: hyperparameters
		:return: Gradient of the negative log likelihood function
		"""
		n_theta = len(theta)
		gradient = []
		Kinv = self.inv_cov_matrix(x,theta)

		for j in range(0,n_theta):
			dKdj = self._d_cov_matrix_d_theta(x,theta,j)
			gradient.append(0.5*tracedot(Kinv,dKdj) - 0.5* dot(t.T,dot(Kinv,dot(dKdj,dot(Kinv,t)))))

		return np.array(gradient)

	def _nll_function(self, x, t):
		"""

		:param x: list of d-dimensional vectors
		:param t: list of responses
		:return: negative log likelihood as function of theta
		"""
		def nll(theta):
			#for p in ltheta:
			#	if p <= 0:
			#		return 1.0e+20
			return self._negativeloglikelihood(x, t, theta)

		return nll

	def _gradient_function(self,x, t):
		"""

		:param x: list of d-dimensional vectors
		:param t: list of responses
		:return: gradient of the negative log likelihood as function of theta
		"""
		def gradient(theta):
			try:
				gr = self._d_nll_d_theta(x,t,theta)
			except np.linalg.linalg.LinAlgError:
				gr = self._d_nll_d_theta(x,t,theta*0.999)
			return gr
		return gradient

	def ml_estimate(self,x,t):
		"""

		:param x: list of d-dimensional vectors
		:param t: list of responses
		:return: maximum likelihood estimate for theta
		"""

		d = len(x[0])
		theta_start = self.get_theta(x, t)
		# print(theta_start)
		func = self._nll_function(x, t)
		fprime = self._gradient_function(x, t)

		#for tnc, l_bfgs_b and slsqp
		# bounds = [(1.0e-15,1e2) for i in range(len(theta_start)) ]
		#for cobyla
		# constr = [(lambda theta : theta[i]) for i in range(len(theta_start)) ]
		bounds = None
		# constr = None

		# from Utilities import minimize
		# try: # Avishai - Added due to errors with the toy example data
		# theta_min = minimize(func,theta_start,bounds,constr,fprime = fprime, method=["simplex"])#all, tnc, l_bfgs_b, cobyla, slsqp, bfgs, powell, cg, simplex or list of some of them
		# except:
		# theta_min = theta_start

		from scipy.optimize import minimize
		best_f=1e6
		for j in range(0, 1):	
			theta0 = theta_start + 20*np.random.random(size=len(theta_start))-10	
			res=minimize(func, theta0, method='l-bfgs-b', bounds=bounds, tol=1e-12, options={'disp':False, 'eps':0.001})
			if res['fun'] < best_f:
				theta_min = res['x']
				best_f = res['fun']
			print "iter: "+str(j) +". Value: " + str(best_f) + ", params: " + str(theta_min)

		return np.array(theta_min)

	#TODO numerical implementation as fallback
	def get_Hessian(self,u,xi, theta):
		"""
		Get the Hessian of the covariance function with respect to u

		:param u: d-dimensional vector
		:param xi: d-dimensional vector
		:param theta: hyperparameters
		:return: Hessian
		"""
		pass
	def get_Jacobian(self,u,xi, theta):
		"""
		Get the Jacobian of the covariance function with respect to u

		:param u: d-dimensional vector
		:param xi: d-dimensional vector
		:param theta: hyperparameters
		:return: Jacobian
		"""
		pass

class GaussianCovariance(Covariance):
	"""
	The classic Gaussian squared exponential covariance function. Suitable to approximate smooth functions.
	"""

	def __call__(self,xi,xj,theta):

		v = np.exp(theta[0])
		vt = np.exp(theta[1])

		w = np.exp(theta[2:])
		#Winv = np.diag(w)

		diff = xi - xj

		#slighly dirty hack to determine whether i==j
		return v * np.exp(-0.5 * (dot(diff.T, w* diff))) + (vt if (xi == xj).all() else 0)

	def get_theta(self,x,t):

		n,d = np.shape(x)
		theta = np.ones(2+d)
		theta[0] = np.log(np.var(t)) if t is not None else 1 #size
		theta[1] = np.log(np.var(t)/4) if t is not None else 1 #noise
		theta[2:] = -2*np.log((np.max(x,0)-np.min(x,0))/2.0)#w # Avishai: Added small number to avoid Inf
		return theta

	def cov_matrix(self,x,theta):
		n, dim = np.shape(x)

		return self.cov_matrix_ij(x,x,theta)

	def cov_matrix_ij(self,xi,xj,theta):
		ni = len(xi)
		nj = len(xj)
		K = np.zeros((ni, nj))

		for i in range(ni):
			for j in range(nj):
				K[i, j] = self(xi[i], xj[j], theta)
		return K

	def _d_cov_d_theta(self,xi,xj,theta,j):

		diff = xi - xj
		v = np.exp(theta[0])
		vt = np.exp(theta[1])

		w = np.exp(theta[2:])
		#Winv = np.diag(w)

		if j == 0:
			return v*np.exp(-0.5 * (dot(diff.T, w* diff)))
		elif j == 1:
			if (xi == xj).all():
				return vt
			else:
				return 0
		else:
			return -0.5 * diff[j-2]**2 * v * np.exp(-0.5 * (dot(diff.T, w* diff))) * w[j-2]
			#0.5*x1**2*exp(-0.5*x3**2/w3 - 0.5*x2**2/w2 - 0.5*x1**2/w1)/w1**2

	def _d_cov_matrix_d_theta(self,x,theta,j):
		vt = np.exp(theta[1])

		n,dim = np.shape(x)
		if j == 1:
			return np.eye(n) *vt
		else:
			return self._d_cov_matrix_d_theta_ij(x,x,theta,j)

	def _d_cov_matrix_d_x(self,x,theta,i,dim,Cov= None):
		"""
		Derive by one dimension of one x
		:param x:
		:param theta:
		:param dim:
		:param Cov: regular covariance Matrix
		:return:
		"""
		#vt = np.exp(theta[1])

		w =np.exp( theta[2:])
		#Winv = np.diag(w)

		n1 = np.shape(x)[0]
		n2 = n1

		x1d = np.atleast_2d(x[:,dim])
		x2d = np.atleast_2d(x[:,dim])

		#diff
		d = np.tile(x1d.T,(1,n2)) - np.tile(x2d,(n1,1))
		if Cov is not None:
			K = -1*d*Cov*w[dim]
		else:
			v = np.exp(theta[0])
			x1 = np.copy(x)
			x2 = np.copy(x)
			x1 = x1 * np.tile(np.sqrt(w),(n1,1))
			x2 = x2 * np.tile(np.sqrt(w),(n2,1))
			K = -2*dot(x1,x2.T)
			K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
			K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))
			K = -1*v*d*np.exp(-0.5*K) * w[dim]

		Res = np.zeros((n1,n2))
		#The ith row contains interactions between  x_i and  x
		Res[i,:] = K[i,:]
		#The ith column contains interactions between  x and  x_i
		Res[:,i] = -K[:,i] # This is different cause x_i is now on the right side of the difference
		Res[i,i] = 0 # the difference between x_i and x_i is always zero
		return Res


	def _d_cov_matrix_d_xi_ij(self,xi,xj,theta,i,dim, Cov=None):
		"""
		Derive by one dimension of one xi

		:param xi:
		:param xj:
		:param theta:
		:param i:
		:param dim:
		:return:
		"""
		#vt = np.exp(theta[1])

		w =np.exp( theta[2:])
		#Winv = np.diag(w)

		n1 = np.shape(xi)[0]
		n2 = np.shape(xj)[0]

		x1d = np.atleast_2d(xi[:,dim])
		x2d = np.atleast_2d(xj[:,dim])

		#diff
		d = np.tile(x1d.T,(1,n2)) - np.tile(x2d,(n1,1))
		if Cov is not None:
			K = -1*d*Cov*w[dim]
		else:
			v = np.exp(theta[0])
			x1 = np.copy(xi)
			x2 = np.copy(xj)
			x1 = x1 * np.tile(np.sqrt(w),(n1,1))
			x2 = x2 * np.tile(np.sqrt(w),(n2,1))
			K = -2*dot(x1,x2.T)
			K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
			K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))

			K = -1*v*d*np.exp(-0.5*K) * w[dim]

		Res = np.zeros((n1,n2))
		#Only the ith row contains interactions between the xi_i and the xj
		Res[i,:] = K[i,:]
		return Res


	def _d_cov_matrix_d_theta_ij(self,xi,xj,theta,j,Cov=None):
		"""

		:param x: list of d-dimensional vectors
		:param theta: hyperparameters
		:return: derivative of the covariance matrix d theta_j
		"""
		n1,dim = np.shape(xi)
		n2 = np.shape(xj)[0]
		w =np.exp( theta[2:])

		if Cov is not None:
			K = Cov
		else:
			v = np.exp(theta[0])
			vt = np.exp(theta[1])

			#Winv = np.diag(w)
			x1 = np.copy(xi)
			x2 = np.copy(xj)

			x1 = x1 * np.tile(np.sqrt(w),(n1,1))
			x2 = x2 * np.tile(np.sqrt(w),(n2,1))
			K = -2*dot(x1,x2.T)
			K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
			K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))
			K = v*np.exp(-0.5*K)

		if j == 0:
			#return np.exp(-0.5 * (dot(diff.T, w* diff)))
			#K = -2*dot(x1,x2.T)
			#K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
			#K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))
			#K = v*np.exp(-0.5*K)
			return K
		elif j == 1:
			return np.zeros((n1,n2))

		else:
			x1j = np.atleast_2d(xi[:,j-2])
			x2j = np.atleast_2d(xj[:,j-2])

			#diff squared
			d = -2 * dot(x1j.T,x2j)
			d += np.tile(x2j*x2j,(n1,1))
			d += np.tile((x1j*x1j).T,(1,n2))

			#K = -2*dot(x1,x2.T)
			#K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
			#K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))

			#K = -0.5*v*d*np.exp(-0.5*K) * w[j-2]
			return -0.5*K*d*w[j-2]


	def get_Hessian(self,u, xi, theta):
		v = np.exp(theta[0])
		vt = np.exp(theta[1])
		w = np.exp(theta[2:])

		Winv = np.diag(w)
		diff = xi - u
		#exp(...) = exp(-1/2*(d1**2/e11 + d2**2/e22 + d3**2/e33)) ;
		expstuff = v * np.exp(-0.5 * (np.dot(diff.T, np.dot(Winv, diff))))

		tile = np.tile(diff*w,(len(u),1))
		hessian = (tile*tile.T - Winv)*expstuff # We assume Winv to be diagonal

		return hessian

	def get_Jacobian(self,u,xi, theta):
		v = np.exp(theta[0])
		vt = np.exp(theta[1])
		w = np.exp(theta[2:])


		Winv = np.diag(w)
		diff = xi - u
		#exp(...) = exp(-1/2*(d1**2/e11 + d2**2/e22 + d3**2/e33)) ;
		expstuff = v * np.exp(-0.5 * (np.dot(diff.T, np.dot(Winv, diff))))

		jacobian = np.atleast_2d(-diff*w*expstuff).T #Eigentlich diff statt -diff weil nach u abgeleitet wird

		return jacobian

