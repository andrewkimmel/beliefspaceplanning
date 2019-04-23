#!/usr/bin/env python

'''
Author: Avishai Sintov
        Rutgers University
        2018-2019
This code is based on the Matlab implementation of the Gaussian Kernel and covariance matrix
'''

import numpy as np
from scipy.linalg import inv

class Covariance(object):

    tinyK = 1.0000e-06
    AlphaHat = None

    def __init__(self, X, Y, theta=None, optimize = True):
        self.Set_Data(X, Y)
        
        if theta is None:
            self.Set_Initial_Theta()
        else:
            self.theta = theta
            self.SigmaLowerBound = np.maximum(1e-2*np.nanstd(self.Y), 1e-3)

        # optimize hyper-parameters
        if optimize:
            self.optimize()
        
        self.K = self.cov_matrix_ij(self.X, self.X)
        # self.K = self.cov_matrix()
        self.Kinv = inv(self.K)   


    def Set_Data(self, X, Y):
        self.X = X
        self.Y = Y
        self.d = self.X.shape[1]
        self.N = self.X.shape[0]

    def Set_Initial_Theta(self):
        tiny = 1e-3

        theta = np.ones(3)# + self.d)
        theta[0] = np.log(np.maximum(np.mean(np.nanstd(self.X, axis=0)), tiny))  #size, signal standard deviation, SigmaL
        theta[1] = np.log(np.maximum(np.nanstd(self.Y)/np.sqrt(2), tiny)) #  characteristic length scale, SigmaF
        theta[2] = np.maximum(np.nanstd(self.Y), tiny) # Sigma0
        
        self.SigmaLowerBound = np.maximum(1e-2*np.nanstd(self.Y), tiny)
        theta[2] = np.log(np.maximum(tiny, theta[2] - self.SigmaLowerBound))

        self.theta = theta
        
        # theta[2:] = -2*np.log((np.max(self.X,0)-np.min(self.X,0)+1e-3)/2.0) # w 

    def _get_sigmaL(self):
        return np.exp(self.theta[0])
    def _get_sigmaF(self):
        return np.exp(self.theta[1])
    def _get_expGamma(self):
        return np.exp(self.theta[2])
    def _get_sigma(self):
        return self.SigmaLowerBound + np.exp(self.theta[2])
    def get_AlphaHat(self):
        if self.AlphaHat is None:
            Theta = np.copy(self.theta)
            while 1:
                try:
                    K = self.cov_matrix_ij(self.X, self.X, Theta)
                    self.L = np.linalg.cholesky(K)
                    self.AlphaHat = np.linalg.inv(self.L.T).dot( np.linalg.inv(self.L).dot(self.Y) )
                    break
                except:
                    Theta = np.copy(self.theta) + (2*np.random.uniform(self.theta.shape) - 1)

        return self.AlphaHat

    # def _get_w(self):
    #     return np.exp(self.theta[2:])

    def _get_theta(self, theta=None):
        if theta is None:
            th = self.theta
        else:
            th = theta
        return np.exp(th[0]), np.exp(th[1]), np.exp(th[2])

    def Gcov(self, xi, xj, theta = None):
        # Computes a scalar covariance of two samples

        if theta is None:
            sigmaL, sigmaF, _ = self._get_theta()
        else:
            sigmaL, sigmaF, _ = self._get_theta(theta)

        sigmaL = np.maximum(sigmaL, self.tinyK)
        sigmaF = np.maximum(sigmaF, self.tinyK)

        diff = xi - xj
        # W = 1. / w

        # slighly dirty hack to determine whether i==j
        # return v * np.exp(-0.5 * (np.dot(diff.T, W* diff))) + (vt if (xi == xj).all() else 0)
        return sigmaF**2 * np.exp(-0.5 * (np.dot(diff.T, diff)) / sigmaL**2) # + (vt if (xi == xj).all() else 0)

    def cov_matrix(self, theta = None):

        # Computes each component individually
        K = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                K[i,j] = self.Gcov(self.X[i,:], self.X[j,:], theta)

        if theta is None:
            sigma2 = (self.SigmaLowerBound + self._get_expGamma())**2
        else:
            sigma2 = (self.SigmaLowerBound + np.exp(theta[2]))**2
        K =  K + sigma2 * np.eye(K.shape[0])

        return K

    def cov_matrix_ij(self, Xi, Xj, theta = None, add_vt = True): 
        # This is more efficient as it computes by matrix multiplication

        if theta is None:
            sigmaL, sigmaF, expGamma = self._get_theta()
        else:
            sigmaL, sigmaF, expGamma = self._get_theta(theta)
        sigmaL = np.maximum(sigmaL, self.tinyK)
        sigmaF = np.maximum(sigmaF, self.tinyK)

        # W = 1. / w

        x1 = np.copy(Xi)
        x2 = np.copy(Xj)
        n1,_ = np.shape(x1)
        n2 = np.shape(x2)[0]
        # x1 = x1 * np.tile(np.sqrt(W),(n1,1))
        # x2 = x2 * np.tile(np.sqrt(W),(n2,1))

        K = -2 * np.dot(x1, x2.T)

        K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
        K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))

        sigma2 = (self.SigmaLowerBound + expGamma)**2
        K = sigmaF**2 * np.exp(-0.5 / sigmaL**2 * K) + (sigma2 * np.eye(K.shape[0]) if add_vt else 0)

        return K

    def optimize(self):
        # bounds = [(-100.,20.) for _ in range(self.d+2)]
        bounds = [(-100.,100.) for _ in range(3)]

        from scipy.optimize import minimize
        res = minimize(self.neg_log_marginal_likelihood, self.theta, method='Nelder-Mead', bounds=bounds, tol=1e-6, options={'disp':False,'eps':1e-10})
        self.theta = res['x']
        self.neg_log_marginal_likelihood_value = res['fun']

        # from Utilities import minimize
        # self.theta = minimize(self.neg_log_marginal_likelihood, self.theta, bounds = bounds, constr = None,fprime = None, method=["l_bfgs_b"]) # all, tnc, l_bfgs_b, cobyla, slsqp, bfgs, powell, cg, simplex or list of some of them

        # print "Optimized hyper-parameters with cost function " + str(res['fun']) + "."
        print "Theta is now " + str(self.theta) + ", " + str(np.exp(self.theta)) + " with cost function " + str(self.neg_log_marginal_likelihood(self.theta)[0][0])

        self.AlphaHat = np.linalg.inv(self.L.T).dot( np.linalg.inv(self.L).dot(self.Y) )
 
    def neg_log_marginal_likelihood(self, theta):
        K = self.cov_matrix_ij(self.X, self.X, theta)
        # K = self.cov_matrix(theta = theta)

        # try:
        #     Kinv = inv(K)
        # except:
        #     Kinv = np.linalg.inv(K)
        # return 0.5*np.dot(self.Y.T, np.dot(Kinv, self.Y)) + 0.5*np.log(np.linalg.det(K)) + 0.5*self.N*np.log(2*np.pi)

        c = 0.5 * self.N * np.log(2*np.pi)
        try:
            self.L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as err:
            if 'Matrix is not positive definite' in str(err):
                return 1e20
            else:
                raise
        
        Linvy = np.dot(np.linalg.inv(self.L), self.Y.reshape(-1,1))
        loglik = 0.5 * Linvy.T.dot(Linvy) + c + np.sum(np.log(np.diag(self.L)))

        return loglik


        
