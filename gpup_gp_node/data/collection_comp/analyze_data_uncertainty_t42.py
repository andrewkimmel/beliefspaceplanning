#!/usr/bin/env python

from scipy.io import loadmat
from sklearn.neighbors import KDTree, NearestNeighbors
import os.path
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var
from gp import GaussianProcess
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/'

Q = {}

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/t42_35_data_discrete_v6_d4_m1.mat' 
Qp = loadmat(file)

print Qp['D'].shape

inx = range(400000)#np.random.choice(Qp['D'].shape[0], 130000, replace=False)

Q['p'] = Qp['D'][inx,:]
Qp = np.delete(Qp['D'], inx, 0)

inx = np.random.choice(Qp.shape[0], 100, replace=False)
# inx2 = np.where(Qp[:,0] > 75.)[0]
# inx2 = inx2[np.random.choice(len(inx2), 50, replace=False)]
# inx = np.concatenate((inx1, inx2), axis=0)
# inx = range(1000,2000) + range(10000,11000) + range(100000,101000) + range(300000,301000) +  range(455000,456000) + range(275030,276030)
# inxp = np.random.choice(len(inx), 500, replace=False)

Qtest = Qp[inx,:]
# Qp = np.delete(Qp, inx, 0)

Xtest = Qtest[:,:6]
Ytest = Qtest[:,6:]


# plt.figure(0)
# for S in Q:
#     plt.plot(Q[S][:,0], Q[S][:,1], '.', label=S)
# plt.plot(Xtest[:,0], Xtest[:,1], '.r', label='test')
# plt.legend()
# # plt.savefig(path + 'data')

# plt.show()
# exit(1)

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

    # for i in range(state_dim):
    #     gp_est = GaussianProcessRegressor(n_restarts_optimizer=9)
    #     gp_est.fit(X_nn[:,:state_dim], Y_nn[:,i])
    #     mm, vv = gp_est.predict(sa[:state_dim].reshape(1,-1), return_std=True)
    #     ds_next[i] = mm
    #     std_next[i] = vv#np.sqrt(np.diag(vv))    

    s_next = sa[:4] + ds_next

    return s_next, std_next

def get_uncertainty_measure(X, Y, X_test_norm, Y_test, mn, mx):
    K = 50
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
        VAR.append(std[:2]**2)

        err = np.linalg.norm(s[:2] - Y_test[i,:2])
        ERR.append(err)

    mean_var = np.mean(np.array(VAR), axis=0)
    max_var = np.max(np.mean(np.array(VAR), axis=1))
    
    return mean_var, max_var, np.mean(ERR), np.max(ERR)


sets = ['p']#, '50', '10']

if 1:
    for Set in sets:
        L = np.concatenate((np.linspace(100, 50000, num = 10), np.linspace(50000, Q[Set].shape[0], num = 10)), axis = 0)
        # L = np.linspace(200, Q[Set].shape[0], num = 10)

        V = []
        ERR = []
        MV = []
        MERR = []
        for l in L:
            print int(l)

            X = Q[Set][:int(l), :6]
            Y = Q[Set][:int(l), 6:]

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

            s, mv, err, merr = get_uncertainty_measure(X, Y, X_test_norm, Ytest, mn, mx)

            V.append(s)
            MV.append(mv)
            ERR.append(err)
            MERR.append(merr)

            with open(path + 'uncertainty_' + Set + '_t42.obj', 'wb') as f:
                pickle.dump([L, V, MV, ERR, MERR], f)           

if 0:  
    L = np.linspace(200, 40000, num = 10)  
    for l in L:
        plt.figure(0)
        plt.clf()
        for Set in sets:
            X = Q[Set][:int(l), :6]

            ax = plt.subplot(1,2,1)
            plt.plot(X[:,0],X[:,1], '.', label=Set)

            ax = plt.subplot(1,2,2)
            plt.plot(X[:,2],X[:,3], '.', label=Set)

        ax = plt.subplot(1,2,1)
        plt.plot(Xtest[:,0], Xtest[:,1], '.r', label='test')
        ax = plt.subplot(1,2,2)
        plt.plot(Xtest[:,2], Xtest[:,3], '.r', label='test')

        plt.legend()
        plt.savefig(path + 'data_L' + str(int(l)))
    exit(1)
        

plt.figure(1, figsize=(12,7))

try:
    Set = sets[0]
    with open(path + 'uncertainty_' + Set + '_t42.obj', 'rb') as f:
        L, V, MV, ERR, MERR = pickle.load(f)
    
    ax = plt.subplot(2,2,1)
    plt.plot(L, V, '.-', label = 'planned')

    ax = plt.subplot(2,2,2)
    plt.plot(L, MV, '.-', label = 'planned')

    ax = plt.subplot(2,2,3)
    plt.plot(L, ERR, '.-', label = 'planned')

    ax = plt.subplot(2,2,4)
    plt.plot(L, MERR, '.-', label = 'planned')
except:
    pass

try:
    Set = sets[2]
    with open(path + 'uncertainty_' + Set + '_t42.obj', 'rb') as f:
        L, V, MV, ERR, MERR = pickle.load(f)
    
    ax = plt.subplot(2,2,1)
    plt.plot(L, V, '.-', label = '10 steps')

    ax = plt.subplot(2,2,2)
    plt.plot(L, MV, '.-', label = '10 steps')

    ax = plt.subplot(2,2,3)
    plt.plot(L, ERR, '.-', label = '10 steps')

    ax = plt.subplot(2,2,4)
    plt.plot(L, MERR, '.-', label = '10 steps')
except:
    pass

try:
    Set = sets[1]
    with open(path + 'uncertainty_' + Set + '_t42.obj', 'rb') as f:
        L, V, MV, ERR, MERR = pickle.load(f)
    
    ax = plt.subplot(2,2,1)
    plt.plot(L, V, '.-', label = '50 steps')

    ax = plt.subplot(2,2,2)
    plt.plot(L, MV, '.-', label = '50 steps')

    ax = plt.subplot(2,2,3)
    plt.plot(L, ERR, '.-', label = '50 steps')

    ax = plt.subplot(2,2,4)
    plt.plot(L, MERR, '.-', label = '50 steps')
except:
    pass

ax = plt.subplot(2,2,1)
plt.title('Variance')
plt.legend()

ax = plt.subplot(2,2,2)
plt.title('Max. variance')
plt.legend()

ax = plt.subplot(2,2,3)
plt.title('MSE')
plt.legend()

ax = plt.subplot(2,2,4)
plt.title('Max. error')
plt.legend()

plt.savefig(path + 'prop_t42')

plt.show()
