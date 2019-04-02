#!/usr/bin/env python

'''
Author: Avishai Sintov
        Rutgers University
        2018-2019
'''

import numpy as np
import time

# algorithm is 'Girard' or 'Matlab'

class GaussianProcess(object):

    def __init__(self, X, Y, optimize = True, theta = None, algorithm = 'Matlab'):

        self.X = X
        self.Y_mean = np.mean(Y)
        self.Y = Y# - self.Y_mean

        self.algorithm = algorithm
        if self.algorithm == 'Girard':
            from covG import Covariance
        elif self.algorithm == 'Matlab':
            from covM import Covariance

        self.cov = Covariance(self.X, self.Y, theta = theta, optimize = optimize)

    def predict(self, x):

        # k_vector = self.cov.cov_matrix_ij(x.reshape(1,-1), self.X) - self.cov._get_vt() # Slower
        k_vector = np.empty((self.X.shape[0],1))
        for i in range(self.X.shape[0]):
            k_vector[i] = self.cov.Gcov(x, self.X[i,:])
        k_vector = k_vector.reshape(1,-1)

        if self.algorithm == 'Girard':
            k = self.cov.Gcov(x, x)

            # Implementation from Girard, Pg. 22
            mean = np.dot(k_vector, np.dot(self.cov.Kinv, self.Y))# + self.Y_mean
            var = k - np.dot(k_vector, np.dot(self.cov.Kinv, k_vector.T))

            # Implementation from Girard, Pg. 32
            # beta = np.dot(self.cov.Kinv, self.Y)
            # mean = 0
            # s = 0
            # for i in range(len(beta)):
            #     mean += beta[i] * k_vector[0][i]
            #     for j in range(len(beta)):
            #         s += self.cov.Kinv[i,j] * k_vector[0][i] * k_vector[0][j]
            # var = k - s
            # mean += self.Y_mean

        elif self.algorithm == 'Matlab':
            # Implementation of Matlab
            AlphaHat = self.cov.get_AlphaHat()
            mean = k_vector.dot(AlphaHat)
            LinvKXXnewc = np.linalg.inv(self.cov.L).dot(k_vector.T)
            diagKNN = self.cov._get_sigmaF()**2 # * np.ones((N,1))
            sigmaHat = self.cov._get_sigma() 
            var = np.maximum(0, sigmaHat**2 + diagKNN - np.sum(np.power(LinvKXXnewc, 2)).T)

        return mean, var

    def batch_predict(self, Xs):

        Xs = np.array(Xs)
        N = Xs.shape[0] # Number of query inputs
        k_vector = self.cov.cov_matrix_ij(Xs, self.X, add_vt = False)

        if self.algorithm == 'Girard':
            # Batch implementation from Girard, Pg. 22
            k = self.cov.cov_matrix_ij(Xs, Xs)
            mean = np.dot(k_vector, np.dot(self.cov.Kinv, self.Y)) # + self.Y_mean #!!!!
            variance = np.diag(np.diag(k)) - np.dot(kv, np.dot(self.cov.Kinv, k_vector.T))# + self.cov._get_vt() #code variance + aleatory variance
        elif self.algorithm == 'Matlab':
            AlphaHat = self.cov.get_AlphaHat()
            mean = k_vector.dot(AlphaHat)
            LinvKXXnewc = np.linalg.inv(self.cov.L).dot(k_vector.T)
            diagKNN = self.cov._get_sigmaF()**2 * np.ones((N,1))
            sigmaHat = self.cov._get_sigma() 
            var = np.maximum(0, sigmaHat**2 + diagKNN - np.sum(np.power(LinvKXXnewc, 2), axis=0).T)

        return mean, var

# if __name__ == '__main__':
#     G = data_load()

