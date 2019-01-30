

import numpy as np
import matplotlib.pyplot as plt


class mean_shift():

    def __init__(self, bandwith=0.01, eps_sqr = .1, plot=False):
        self.bandwith = bandwith
        self.eps_sqr = eps_sqr
        self.plot = plot

    def gaussian_kernel(self, xi, xj):
        # Computes a scalar covariance of two samples

        diff = xi - xj

        return np.exp(- 0.5 * self.bandwith * (np.dot(diff.T, diff)))

    def get_mean_shift(self, X):

        m = np.mean(X, 0)

        if self.plot:
            plt.plot(X[:,0], X[:,1], '.k')
            plt.plot(m[0], m[1], '*r')

        while 1:
            m_prev = np.copy(m)

            m = self.shift_point(m, X)

            if self.plot:
                plt.plot(m[0], m[1], '.m')

            if np.linalg.norm(m_prev-m) < self.eps_sqr:
                break

        if self.plot:
            plt.plot(m[0], m[1], '*g')
            plt.show()

        return m
        

    def shift_point(self, m, X):
        m_next = np.zeros(len(m))
        total_weight = 0
        for i in range(X.shape[0]):
            w = self.gaussian_kernel(m, X[i])

            m_next += X[i] * w
            total_weight += w

        return m_next / total_weight




        