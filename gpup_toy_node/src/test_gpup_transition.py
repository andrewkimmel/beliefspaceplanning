#!/usr/bin/env python

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_toy_node/gpuppy/')

import numpy as np
from GaussianProcess import GaussianProcess
from Covariance import GaussianCovariance, Covariance
from UncertaintyPropagation import UncertaintyPropagationApprox, UncertaintyPropagationExact, UncertaintyPropagationMC
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn

import GPy

file_name = '/home/pracsys/catkin_ws/src/beliefspaceplanning/toy_simulator/data/transition_data_cont.obj'

print('Loading data from ' + file_name)
with open(file_name, 'rb') as filehandler:
    data = pickle.load(filehandler)
print('Loaded transition data of size %d.'%len(data))

x_data = np.array([np.concatenate((item[0],item[1]), axis=0) for item in data])
y_data = np.array([item[2] for item in data])
# plt.plot(x_data[:,0], x_data[:,1], '+y')
# plt.show()

x_test = x_data[:10,:]
y_test = y_data[:10,:]
x_data = x_data[10:,:]
y_data = y_data[10:,:]

Dx = x_data.shape[1]
Dy = y_data.shape[1]

kdt = KDTree(x_data, leaf_size=10, metric='euclidean')
K = 100
K_up = 100
K_many = 500

def predict(sa, X, Y):
    idx = kdt.query(sa, k=K, return_distance=False)
    X_nn = X[idx,:].reshape(K, Dx)
    Y_nn = Y[idx,:].reshape(K, Dy)

    d = Y_nn.shape[1]
    m = np.empty(d)
    s = np.empty(d)
    for i in range(d):
        gp_est = GaussianProcess(X_nn, Y_nn[:,i], GaussianCovariance())
        m[i], s[i] = gp_est.estimate(sa[0])
    return m, s

def predict_many(SA, X, Y): # Assume that the test points are near each other (Gaussian distributed...)
    sa = np.mean(SA, 0).reshape(1,-1)
    idx = kdt.query(sa, k=K_many, return_distance=False)
    X_nn = X[idx,:].reshape(K_many, Dx)
    Y_nn = Y[idx,:].reshape(K_many, Dy)

    m = np.empty([SA.shape[0], Dy])
    s = np.empty([SA.shape[0], Dy])
    for i in range(Dy):
        gp_est = GaussianProcess(X_nn, Y_nn[:,i], GaussianCovariance())
        m[:,i], s[:,i] = gp_est.estimate_many(SA)
    return m, s

def UP(sa_mean, sa_Sigma, X, Y):
    idx = kdt.query(sa, k=K_up, return_distance=False)
    X_nn = X[idx,:].reshape(K_up, Dx)
    Y_nn = Y[idx,:].reshape(K_up, Dy)

    d = Y_nn.shape[1]
    m = np.empty(d)
    s = np.empty(d)
    for i in range(d):
        gp_est = GaussianProcess(X_nn, Y_nn[:,i], GaussianCovariance())
        up = UncertaintyPropagationExact(gp_est)
        m[i], s[i] = up.propagate_GA(sa_mean, sa_Sigma)
    return m, s

def GPy_predict(sa, X, Y):
    idx = kdt.query(sa, k=K, return_distance=False)
    X_nn = X[idx,:].reshape(K, Dx)
    Y_nn = Y[idx,:].reshape(K, Dy)

    kernel = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=0.5)

    d = Y_nn.shape[1]
    m = np.empty(d)
    s = np.empty(d)
    for i in range(d):
        gpy = GPy.models.GPRegression(X_nn, Y_nn[:,i].reshape(-1,1), kernel)
        gpy.optimize(messages=False)
        m[i], s[i] = gpy.predict(sa[0].reshape(1,-1))
    return m, s


ijx = 6

# for ijx in range(1):

sa = x_test[ijx,:].reshape(1,-1)

m_gp, s_gp = predict(sa, x_data, y_data)
s_gp = np.sqrt(s_gp)

print "----------- GP validation -----------"
print "test point: ", sa
print "m_gp: ", m_gp
print "s_gp: ", s_gp
print "m_real: ", y_test[ijx,:]
print "Error: ", np.linalg.norm(m_gp-y_test[ijx,:])
# exit(1)
print "---------------- UP ----------------"

mean = sa.reshape((-1,)) # The mean of a normal distribution
Sigma = np.diag([0.06**2, 0.06**2, 0.06**2, 0.06**2]) # The covariance matrix (must be diagonal because of lazy programming)

m, s = UP(mean, Sigma, x_data, y_data)
s = np.sqrt(s)

print "m_gpup: ", m
print "s_gpup: ", s

print "-------- Particles --------------------------"

N = int(1e1)
X_belief = np.array([np.diag(np.random.normal(mean, np.sqrt(Sigma))) for _ in range(N)]).reshape(N,Dx) #
# X_belief = np.tile(sa, (N,1))

Y_belief_m = []
for i in range(N):
    # print i
    mean_b, variance_b = predict(X_belief[i,:].reshape(1,-1), x_data, y_data)
    variance_b[0] = 0 if np.absolute(variance_b[0]) <= 1e-3 else variance_b[0]
    variance_b[1] = 0 if np.absolute(variance_b[1]) <= 1e-3 else variance_b[1]
    if variance_b[0] < 0 or variance_b[1] < 0:
        continue
    Y_belief_m.append([np.random.normal(mean_b[0], np.sqrt(variance_b[0])), np.random.normal(mean_b[1], np.sqrt(variance_b[1]))])
Y_belief_m = np.array(Y_belief_m)
Y_belief_mean_m = np.mean(Y_belief_m,0)
print "Particle mean m: ", Y_belief_mean_m
print "Particle std m: ", np.std(Y_belief_m,0)


means_b, variances_b = predict_many(X_belief, x_data, y_data)
# print np.concatenate((means_b, variances_b), axis=1)
Y_belief = []
for i in range(N):
    variances_b[i,0] = 0 if np.absolute(variances_b[i,0]) <= 1e-3 else variances_b[i,0]
    variances_b[i,1] = 0 if np.absolute(variances_b[i,1]) <= 1e-3 else variances_b[i,1]
    if variances_b[i,0] < 0 or variances_b[i,1] < 0:
        continue
    Y_belief.append([np.random.normal(means_b[i,0], np.sqrt(variances_b[i,0])), np.random.normal(means_b[i,1], np.sqrt(variances_b[i,1]))])
Y_belief = np.array(Y_belief)
Y_belief_mean = np.mean(Y_belief,0)
print "Particle mean: ", Y_belief_mean
print "Particle std: ", np.std(Y_belief,0)

print "-------- GPy --------------------------"

# Y_belief_g = []
# for i in range(N):
#     # print i
#     mean_g, variance_g = GPy_predict(X_belief[i,:].reshape(1,-1), x_data, y_data)
#     variance_g[0] = 0 if np.absolute(variance_g[0]) <= 1e-3 else variance_g[0]
#     variance_g[1] = 0 if np.absolute(variance_g[1]) <= 1e-3 else variance_g[1]
#     if variance_g[0] < 0 or variance_g[1] < 0:
#         continue
#     Y_belief_g.append([np.random.normal(mean_g[0], np.sqrt(variance_g[0])), np.random.normal(mean_g[1], np.sqrt(variance_g[1]))])
# Y_belief_g = np.array(Y_belief_g)
# Y_belief_mean_g = np.mean(Y_belief_g,0)
# print "Particle mean g: ", Y_belief_mean_g
# print "Particle std g: ", np.std(Y_belief_g,0)

print "-------------------------------------"


#####
plt.plot(X_belief[:,0], X_belief[:,1], '.c')
plt.plot(Y_belief[:,0], Y_belief[:,1], '.k', label='propagated particles')
plt.plot(Y_belief_m[:,0], Y_belief_m[:,1], '.m', label='propagated particles - individual')
# plt.plot(Y_belief_g[:,0], Y_belief_g[:,1], 'pc', label='propagated particles GPy')


plt.plot(sa[0][0], sa[0][1], 'sr', label='test point')
plt.plot(m_gp[0], m_gp[1], 'ob', label='GP prediction')
plt.plot(y_test[ijx,0], y_test[ijx,1], '*y', label='real next point')

plt.errorbar(m[0], m[1], xerr=s[0], yerr=s[1], ecolor='m', label='GPUP std.')
plt.errorbar(m_gp[0], m_gp[1], xerr=s_gp[0], yerr=s_gp[1], ecolor='b', label='GP std.')

plt.plot(m[0], m[1], 'pm', label='GPUP mean prediction')

# plt.plot(Y_belief_mean_g[0], Y_belief_mean_g[1], '*m', label='particles mean_g')
plt.plot(Y_belief_mean_m[0], Y_belief_mean_m[1], '*m', label='particles mean_m')
plt.plot(Y_belief_mean[0], Y_belief_mean[1], '+k', label='particles mean')


plt.legend()

plt.show()