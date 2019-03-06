#!/usr/bin/env python

from scipy.io import loadmat
from sklearn.neighbors import KDTree, NearestNeighbors
import os.path
import pickle
import matplotlib.pyplot as plt
import var
from gp import GaussianProcess
import numpy as np

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_discrete_v13_d4_m10.mat' 
Q = loadmat(file)
Q = Q['D']

inx = np.random.choice(Q.shape[0], 100, replace=False)
Qtest = Q[inx,:]
Q = np.delete(Q, inx, 0)

Xtest = Qtest[:,:6]
Ytest = Qtest[:,6:]


# ---------------------------------------------------------------

def normz(x, x_min_X, x_max_X):
    d = len(x)
    return (x-x_min_X[:d])/(x_max_X[:d]-x_min_X[:d])

def denormz(x, x_min_X, x_max_X):
    d = len(x)
    return  x*(x_max_X[:d]-x_min_X[:d])+x_min_X[:d]

def normz_change(dx, x_min_X, x_max_X):
    d = len(dx)
    return dx/(x_max_X[:d]-x_min_X[:d])

def denormz_change(dx, x_min_X, x_max_X):
    d = len(dx)
    return  dx*(x_max_X[:d]-x_min_X[:d])

def one_predict(sa, X, Y, kdt, K = 100):
    state_dim = 4

    idx = kdt.query(sa.reshape(1,-1), k = K, return_distance=False)
    X_nn = X[idx,:].reshape(K, state_dim + 2)
    Y_nn = Y[idx,:].reshape(K, state_dim)

    ds_next = np.zeros((state_dim,))
    std_next = np.zeros((state_dim,))
    try:
        for i in range(state_dim):
            if i == 0:
                gp_est = GaussianProcess(X_nn[:,:state_dim], Y_nn[:,i], optimize = True, theta = None)
                Theta = gp_est.cov.theta
            else:
                gp_est = GaussianProcess(X_nn[:,:state_dim], Y_nn[:,i], optimize = False, theta = Theta)
            mm, vv = gp_est.predict(sa[:state_dim])
            ds_next[i] = mm
            std_next[i] = np.sqrt(np.diag(vv))
    except:
        print "pass"
        return np.array([-1,-1,-1,-1]), np.array([-1,-1,-1,-1])

    s_next = sa[:4] + ds_next

    return s_next, std_next

def get_sample(X, neigh, mn, mx):
    A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])

    n = 0
    s = np.array([0,0,0,0])
    while n < 1:
        for i in range(4):
            s[i] = np.random.uniform(mn[i], mx[i], 1)
        a = A[np.random.randint(A.shape[0])]
        sa = np.concatenate((s,a))
        sa = normz(sa, mn, mx)
        rng = neigh.radius_neighbors(sa.reshape(1, -1))

        n = np.array(rng[1][0]).shape[0]

    return sa

def get_uncertainty_measure(X, Y, X_test_norm, Y_test, mn, mx):
    K = 100
    r = 0.3
    kdt = KDTree(X, leaf_size=100, metric='euclidean')
    neigh = NearestNeighbors(K, r)
    neigh.fit(X)  

    VAR = []
    ERR = []
    for i in range(X_test_norm.shape[0]):
        print "Step %d..."%i
        s = np.array([-1,-1,-1,-1])
        while np.all(s == np.array([-1,-1,-1,-1])):
            sa = X_test_norm[i,:]
            s, std = one_predict(sa, X, Y, kdt, K = K)
            s = denormz(s, mn, mx)
            std = denormz_change(std, mn, mx)
        VAR.append(std**2)

        err = np.linalg.norm(s - Y_test[i,:])
        ERR.append(err)

    mean_var = np.mean(np.array(VAR), axis=0)
    
    return mean_var, np.mean(ERR)



L = np.linspace(1000, Q.shape[0], num = 20)

V = []
ERR = []
for l in L:
    print l

    X = Q[:int(l), :6]
    Y = Q[:int(l), 6:]

    mx = np.max(X, 0)
    mn = np.min(X, 0)

    X_test_norm = np.copy(Xtest)
    Y_test_norm = np.copy(Ytest)
    for i in range(4+2):
        X[:,i] = (X[:,i]-mn[i])/(mx[i]-mn[i])
        X_test_norm[:,i] = (Xtest[:,i]-mn[i])/(mx[i]-mn[i])
    for i in range(4):
        Y[:,i] = (Y[:,i]-mn[i])/(mx[i]-mn[i])
        Y_test_norm[:,i] = (Ytest[:,i]-mn[i])/(mx[i]-mn[i])

    Y -= X[:,:4]
    Y_test_norm -= X_test_norm[:,:4]

    s, err = get_uncertainty_measure(X, Y, X_test_norm, Ytest, mn, mx)

    V.append(s)
    ERR.append(err)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
with open(path + 'uncertainty.obj', 'wb') as f:
    pickle.dump([L, V, ERR], f)

figure(1)
plt.plot(L, V, '-')

figure(2)
plt.plot(L, ERR, '-')

plt.show()
