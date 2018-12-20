#!/usr/bin/env python

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_toy_node/gpuppy/')

import numpy as np
from GaussianProcess import GaussianProcess
from Covariance import GaussianCovariance
from UncertaintyPropagation import UncertaintyPropagationApprox, UncertaintyPropagationExact
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn

file_name = '/home/pracsys/catkin_ws/src/beliefspaceplanning/toy_simulator/data/transition_data_discrete.obj'

print('Loading data from ' + file_name)
with open(file_name, 'rb') as filehandler:
    data = pickle.load(filehandler)
print('Loaded transition data of size %d.'%len(data))

ax1 = plt.subplot2grid((3, 5), (0, 2), colspan=3, rowspan=2)

x_data = np.array([np.concatenate((item[0],item[1]), axis=0) for item in data])
y_data = np.array([item[2] for item in data])
plt.plot(y_data[:,0], y_data[:,1], '+k')

x_test = x_data[:10,:]
y_test = y_data[:10,:]
x_data = x_data[10:,:]
y_data = y_data[10:,:]

kdt = KDTree(x_data, leaf_size=10, metric='euclidean')
K = 100
K_up = 100

def predict(sa, X, Y):
    idx = kdt.query(sa, k=K, return_distance=False)
    X_nn = X[idx,:].reshape(K, 4)
    Y_nn = Y[idx,:].reshape(K, 2)

    d = Y_nn.shape[1]
    m = np.empty(d)
    s = np.empty(d)
    for i in range(d):
        gp_est = GaussianProcess(X_nn, Y_nn[:,i], GaussianCovariance())
        m[i], s[i] = gp_est.estimate(sa[0])
    return m, s

def UP(sa_mean, sa_Sigma, X, Y):
    idx = kdt.query(sa, k=K_up, return_distance=False)
    X_nn = X[idx,:].reshape(K_up, 4)
    Y_nn = Y[idx,:].reshape(K_up, 2)

    d = Y_nn.shape[1]
    m = np.empty(d)
    s = np.empty(d)
    for i in range(d):
        gp_est = GaussianProcess(X_nn, Y_nn[:,i], GaussianCovariance())
        up = UncertaintyPropagationApprox(gp_est)
        print up.propagate_GA(sa_mean, sa_Sigma)
    # return m, s


print "---------- GP validation ------------"

sa = x_test[9,:].reshape(1,-1)

print predict(sa, x_data, y_data)
print sa, y_test[9,:]


print "---------- UP ------------"

mean = sa.reshape(-1,1) # The mean of a normal distribution
Sigma = np.diag([0.05**2, 0.05**2, 0.01**2, 0.01**2]) # The covariance matrix (must be diagonal because of lazy programming)

UP(mean, Sigma, x_data, y_data)

exit(1)

print "----------------------------------------------"


N = int(1e4)
X_belief = np.array([np.random.normal(mean, np.sqrt(Sigma)) for _ in range(N)]).reshape(N,1) #
ax4 = plt.subplot2grid((3, 5), (2, 2), colspan=3, rowspan=1)
plt.plot(X_belief, np.tile(0., N), '.k')
x = np.linspace(0, 6, 1000).reshape(-1,1)
plt.plot(x,mlab.normpdf(x, mean, np.sqrt(Sigma)))
plt.xlabel('x')


ax2 = plt.subplot2grid((3, 5), (0, 0), colspan=1, rowspan=2)
means_b, variances_b = gp_est.estimate_many(X_belief)
Y_belief = np.array([np.random.normal(means_b[i], np.sqrt(variances_b[i])) for i in range(N)]).reshape(N,1) #
plt.plot(np.tile(0., N), Y_belief, '.k')
plt.ylabel('p(y)')

ylim = ax1.get_ylim()
mu_Y = np.mean(Y_belief)
sigma2_Y = np.std(Y_belief)
y = np.linspace(ylim[0], ylim[1], 1000).reshape(-1,1)
plt.plot(mlab.normpdf(y, mu_Y, sigma2_Y), y, '-b')
plt.plot(mlab.normpdf(y, out_mean, np.sqrt(out_variance)), y, ':r')

ax3 = plt.subplot2grid((3, 5), (0, 1), rowspan=2)
plt.hist(means_b, bins=20, orientation='horizontal')

ax2.set_ylim(ylim)
ax3.set_ylim(ylim)

plt.show()