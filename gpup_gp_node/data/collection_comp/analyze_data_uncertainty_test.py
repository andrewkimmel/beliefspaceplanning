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
from matplotlib.patches import Ellipse

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# import GPy

np.random.seed(1)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/'

Q = {}

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_discrete_v13_d4_m1.mat' 
Qp = loadmat(file)

inx = range(200000)#np.random.choice(Qp['D'].shape[0], 130000, replace=False)

mx = np.max(Qp['D'][:,:6], 0)
mn = np.min(Qp['D'][:,:6], 0)

Q['p'] = Qp['D'][inx,:]
Qp = np.delete(Qp['D'], inx, 0)

# inx1 = np.random.choice(Qp.shape[0], 50, replace=False)
inx2 = np.where(Qp[:,0] > 75.)[0]
inx2 = inx2[np.random.choice(len(inx2), 1, replace=False)]
inx = inx2#np.concatenate((inx1, inx2), axis=0)
# inx = range(1000,2000) + range(10000,11000) + range(100000,101000) + range(300000,301000) +  range(455000,456000) + range(275030,276030)
# inxp = np.random.choice(len(inx), 500, replace=False)

Qtest = Qp[inx,:]
# Qp = np.delete(Qp, inx, 0)

l = 2000

Xtest = Qtest[:,:6]
Ytest = Qtest[:,6:]

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/sim_data_discrete_v100_d4_m1_random10A.mat' 
Q10 = loadmat(file)
Q['10'] = Q10['D'][:l,:]

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/sim_data_discrete_v100_d4_m1_random50A.mat' 
Q50 = loadmat(file)
Q['50'] = Q50['D'][:l,:]

plt.figure(0)
# ax = plt.subplot(1,2,1)
# for S in Q:
#     plt.plot(Q[S][:,0], Q[S][:,1], '.', label=S)
# plt.plot(Xtest[:,0], Xtest[:,1], '.r', label='test')

# ax = plt.subplot(1,2,2)
# for S in Q:
#     plt.plot(Q[S][:,2], Q[S][:,3], '.', label=S)
# plt.plot(Xtest[:,2], Xtest[:,3], '.r', label='test')
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

    Theta = np.array([-14.15405746, -15.54035183,   8.26089476,   9.69821847,   8.35259586,   7.62647084])

    idx = kdt.query(sa.reshape(1,-1), k = K, return_distance=False)
    X_nn = X[idx,:].reshape(K, state_dim + 2)
    Y_nn = Y[idx,:].reshape(K, state_dim)

    ds_next = np.zeros((state_dim,))
    std_next = np.zeros((state_dim,))
    try:
        for i in range(state_dim):
            if i == 0:
                gp_est = GaussianProcess(X_nn[:,:state_dim], Y_nn[:,i], optimize = False, theta = Theta)
                # Theta = gp_est.cov.theta
            else:
                gp_est = GaussianProcess(X_nn[:,:state_dim], Y_nn[:,i], optimize = False, theta = Theta)
            mm, vv = gp_est.predict(sa[:state_dim])
            ds_next[i] = mm
            std_next[i] = np.sqrt(np.diag(vv))
    except:
        print "pass"
        return np.array([-1,-1,-1,-1]), np.array([-1,-1,-1,-1]), []
    
    # SKlearn
    # for i in range(state_dim):
    #     gp_est = GaussianProcessRegressor(n_restarts_optimizer=9)
    #     gp_est.fit(X_nn[:,:state_dim], Y_nn[:,i])
    #     mm, vv = gp_est.predict(sa[:state_dim].reshape(1,-1), return_std=True)
    #     ds_next[i] = mm
    #     std_next[i] = vv#np.sqrt(np.diag(vv))    

    # GPy
    # for i in range(state_dim):
    #     kernel = GPy.kern.RBF(input_dim=state_dim, variance=1., lengthscale=1.)
    #     gp_est = GPy.models.GPRegression(X_nn[:,:state_dim], Y_nn[:,i].reshape(-1,1), kernel)

    #     gp_est.optimize(messages=False)
    #     # m.optimize_restarts(num_restarts = 10)

    #     mm, vv = gp_est.predict(sa[:state_dim].reshape(1,state_dim))
    #     ds_next[i] = mm
    #     std_next[i] = np.sqrt(np.diag(vv))

    s_next = sa[:4] + ds_next

    print sa[:4], s_next

    return s_next, std_next, X_nn

Set = '10'

X = np.copy(Q[Set][:int(l), :6])
Y = np.copy(Q[Set][:int(l), 6:])

Xtest = np.concatenate((Xtest, X[5,:].reshape(1,-1)),axis=0)
X = np.delete(X, 5, 0)
Ytest = np.concatenate((Ytest, Y[5,:].reshape(1,-1)),axis=0)
Y = np.delete(Y, 5, 0)

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

K = 50
kdt = KDTree(X, leaf_size=100, metric='euclidean')

# plt.figure(1)
ax = plt.subplot(1,1,1)
plt.plot(Q[Set][:int(l),0], Q[Set][:int(l),1], '.y', label=Set)
# plt.plot(X[:int(l),0], X[:int(l),1], '.y', label=Set)
# ax = plt.subplot(1,2,2)
# plt.plot(Q[Set][:int(l),2], Q[Set][:int(l),3], '.y', label=Set)

ST = []
for i in range(X_test_norm.shape[0]):
    print "Step %d..."%i
    s = np.array([-1,-1,-1,-1])
    while np.all(s == np.array([-1,-1,-1,-1])):
        sa = X_test_norm[i,:]
        s, std, X_nn = one_predict(sa, X, Y, kdt, K = K)
        s = denormz(s, mn, mx)
        std = denormz_change(std, mn, mx)

        # X_nn = denormz(X_nn, mn, mx)

    print std
    print np.linalg.norm(s[:2]-(X_test_norm[i,:2] + Y_test_norm[i,:2]))*1000
    ST.append(np.linalg.norm(s[:2]-(X_test_norm[i,:2] + Y_test_norm[i,:2]))*1000)

    ax = plt.subplot(1,1,1)
    plt.plot(Xtest[i,0], Xtest[i,1], '.r')
    # plt.plot(X_test_norm[i,0], X_test_norm[i,1], '.r')
    plt.plot(s[0], s[1], 'om')
    plt.plot(Ytest[i,0], Ytest[i,1], 'xg')
    # plt.plot(X_test_norm[i,0] + Y_test_norm[i,0], X_test_norm[i,1] + Y_test_norm[i,1], 'xg')
    plt.plot([s[0], Xtest[i,0]], [s[1], Xtest[i,1]], '-k')
    # plt.plot([s[0], X_test_norm[i,0]], [s[1], X_test_norm[i,1]], '-k')
    # plt.plot(X_nn[:,0], X_nn[:,1],'.k')
    ell = Ellipse((s[0], s[1]), std[0], std[1], 0)
    ax.add_artist(ell)

# print np.mean(ST), np.std(ST)



    # ax = plt.subplot(1,2,2)
    # plt.plot(Xtest[i,2], Xtest[i,3], '.r')
    # plt.plot(s[2], s[3], '.m')
    # plt.plot(Ytest[i,2], Ytest[i,3], 'xg')
    # plt.plot([s[2], Xtest[i,2]], [s[3], Xtest[i,3]], '-k')
    # plt.plot(X_nn[:,2], X_nn[:,3],'.k')
    # ell = Ellipse((s[2], s[3]), std[2], std[3], 0)
    # ax.add_artist(ell)




# def func(x, v = 0.3):
#     return x*np.sin(x)+np.random.normal(0, v)
#     # return 3*x+4+np.random.normal(0, 0.01)

# # x_data = np.linspace(0, 4, 5).reshape(-1,1)
# x_data = np.random.uniform(0, 5, 7).reshape(-1,1)
# y_data = np.array([func(i,0.2) for i in x_data]) #

# x_real = np.linspace(0, 20, 100).reshape(-1,1)
# y_real = np.array([func(i, 0) for i in x_real]) 

# gp_est = GaussianProcess(x_data, y_data.reshape((-1,)), optimize = True, theta=None)
# # kernel = GPy.kern.RBF(input_dim=1, variance=0.9, lengthscale=0.5)
# # gp_est = GPy.models.GPRegression(x_data, y_data, kernel)
# # gp_est.optimize()

# x_new = np.linspace(0, 20, 100).reshape(-1,1)
# means = np.empty(100)
# variances = np.empty(100)
# for i in range(100):
#     means[i], variances[i] = gp_est.predict(x_new[i].reshape(-1,1))

# plt.plot(x_real, y_real)
# plt.plot(x_data, y_data,'xk')
# msl = (means.reshape(1,-1)[0]-np.sqrt(variances))#.reshape(-1,1)
# msu = (means.reshape(1,-1)[0]+np.sqrt(variances))#.reshape(-1,1)[0]
# plt.plot(x_new, means,'-r')
# plt.fill_between(x_new.reshape(1,-1)[0], msl, msu)
plt.show()






    




        
