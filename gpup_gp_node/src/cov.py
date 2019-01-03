#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv


class Covariance(object):

    def __init__(self, X, Y, theta=None, optimize = True):
        self.Set_Data(X, Y)
        
        if theta is None:
            self.Set_Initial_Theta()
        else:
            self.theta = theta

        # optimize hyper-parameters
        if optimize:
            self.optimize()
        
        self.K = self.cov_matrix_ij(self.X, self.X)
        # print self.K
        # self.K = self.cov_matrix()
        self.Kinv = inv(self.K)   


    def Set_Data(self, X, Y):
        self.X = X
        self.Y = Y
        self.d = self.X.shape[1]
        self.N = self.X.shape[0]

    def Set_Initial_Theta(self):

        theta = np.ones(2 + self.d)
        theta[0] = np.log(np.var(self.Y))  #size
        theta[1] = np.log(np.var(self.Y)/4)  #noise
        theta[2:] = -2*np.log((np.max(self.X,0)-np.min(self.X,0)+1e-3)/2.0) # w 
        self.theta = theta

    def _get_v(self):
        return np.exp(self.theta[0])
    def _get_vt(self):
        return np.exp(self.theta[1])
    def _get_w(self):
        return np.exp(self.theta[2:])

    def _get_theta(self, theta=None):
        if theta is None:
            th = self.theta
        else:
            th = theta
        return np.exp(th[0]), np.exp(th[1]), np.exp(th[2:])


    def Gcov(self,xi,xj):
        # Computes a scalar covariance of two samples

        v, vt, w = self._get_theta()

        diff = xi - xj
        W = 1. / w

        #slighly dirty hack to determine whether i==j
        return v * np.exp(-0.5 * (np.dot(diff.T, W* diff))) + (vt if (xi == xj).all() else 0)

    def cov_matrix(self):
        vt = self._get_vt()

        # Computes each component indivisually
        K = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                K[i,j] = self.Gcov(self.X[i,:], self.X[j,:])

        return K + vt*np.eye(K.shape[0])

    def cov_matrix_ij(self, Xi, Xj, theta = None, add_vt = True): 
        # This is more efficient as it computes by matrix multiplication

        if theta is None:
            v, vt, w = self._get_theta()
        else:
            v, vt, w = self._get_theta(theta)
        
        W = 1. / w

        x1 = np.copy(Xi)
        x2 = np.copy(Xj)
        n1,_ = np.shape(x1)
        n2 = np.shape(x2)[0]
        x1 = x1 * np.tile(np.sqrt(W),(n1,1))
        x2 = x2 * np.tile(np.sqrt(W),(n2,1))

        K = -2 * np.dot(x1, x2.T)

        K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
        K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))

        K = v*np.exp(-0.5*K) + (vt*np.eye(K.shape[0]) if add_vt else 0)

        return K

    def optimize(self):
        bounds = [(-100.,20.) for _ in range(self.d+2)]
        # res = minimize(self.neg_log_marginal_likelihood, self.theta, method='l-bfgs-b', bounds=bounds,tol=1e-20, options={'disp':False,'eps':1e-10})
        # self.theta = res['x']

        from Utilities import minimize
        self.theta = minimize(self.neg_log_marginal_likelihood, self.theta, bounds = bounds, constr = None,fprime = None, method=["l_bfgs_b"])#all, tnc, l_bfgs_b, cobyla, slsqp, bfgs, powell, cg, simplex or list of some of them

        # print "Optimized hyper-parameters with cost function " + str(res['fun']) + "."
        print "Theta is now " + str(self.theta)

    def neg_log_marginal_likelihood(self, theta):
        K = self.cov_matrix_ij(self.X, self.X, theta)
        
        try:
            Kinv = inv(K)
        except:
            Kinv = np.linalg.inv(K)

        return 0.5*np.dot(self.Y.T, np.dot(Kinv, self.Y)) + 0.5*np.log(np.linalg.det(K)) + 0.5*self.N*np.log(2*np.pi)

