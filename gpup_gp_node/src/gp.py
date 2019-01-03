#!/usr/bin/env python

import numpy as np
from cov import Covariance
import time

class GaussianProcess(object):

    def __init__(self, X, Y, optimize = True, theta = None):

        self.X = X
        self.Y_mean = np.mean(Y)
        self.Y = Y - self.Y_mean

        self.cov = Covariance(self.X, self.Y, theta = theta, optimize = optimize)

    def predict(self, x):

        k = self.cov.Gcov(x, x)
        # k_vector = self.cov.cov_matrix_ij(x.reshape(1,-1), self.X) - self.cov._get_vt() # Slower

        k_vector = np.empty((self.X.shape[0],1))
        for i in range(self.X.shape[0]):
            k_vector[i] = self.cov.Gcov(x, self.X[i,:])
        k_vector = k_vector.reshape(1,-1)

        # Implementation from Girard, Pg. 22
        mean = np.dot(k_vector, np.dot(self.cov.Kinv, self.Y)) + self.Y_mean
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

        return mean, var

    def batch_predict(self, Xs):

        Xs = np.array(Xs)
        k = self.cov.cov_matrix_ij(Xs, Xs)
        kv = self.cov.cov_matrix_ij(Xs, self.X, add_vt = False)
        mean = np.dot(kv, np.dot(self.cov.Kinv, self.Y)) + self.Y_mean
        variance = np.diag(np.diag(k)) - np.dot(kv, np.dot(self.cov.Kinv, kv.T))# + self.cov._get_vt() #code variance + aleatory variance

        return mean, variance



if __name__ == '__main__':
    G = data_load()

