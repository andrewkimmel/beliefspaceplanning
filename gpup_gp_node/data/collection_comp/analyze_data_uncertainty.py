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

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/'

Q = {}

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_discrete_v13_d4_m1.mat' 
Qp = loadmat(file)

inx = range(200000)#np.random.choice(Qp['D'].shape[0], 130000, replace=False)

Q['p'] = Qp['D'][inx,:]
Qp = np.delete(Qp['D'], inx, 0)

inx = np.random.choice(Qp.shape[0], 500, replace=False)
# inx2 = np.where(Qp[:,0] > 75.)[0]
# inx2 = inx2[np.random.choice(len(inx2), 50, replace=False)]
# inx = np.concatenate((inx1, inx2), axis=0)
# inx = range(1000,2000) + range(10000,11000) + range(100000,101000) + range(300000,301000) +  range(455000,456000) + range(275030,276030)
# inxp = np.random.choice(len(inx), 500, replace=False)

Qtest = Qp[inx,:]
# Qp = np.delete(Qp, inx, 0)

Xtest = Qtest[:,:6]
Ytest = Qtest[:,6:]

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/sim_data_discrete_v100_d4_m1_random10A.mat' 
Q10 = loadmat(file)
Q['10'] = Q10['D']

file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/sim_data_discrete_v100_d4_m1_random50A.mat' 
Q50 = loadmat(file)
Q['50'] = Q50['D']

M = np.min([Q[Set].shape[0] for Set in Q.keys()])

# plt.figure(0)
# for S in Q:
#     if S == '50':
#         continue
#     plt.plot(Q[S][:150000,0], Q[S][:150000,1], '.', label=S)
# # plt.plot(Xtest[:,0], Xtest[:,1], '.r', label='test')
# plt.legend()
# plt.savefig(path + 'data')

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


sets = Q.keys()

if 0:
    for Set in sets:
        # L = np.concatenate((np.linspace(51, 100000, num = 10), np.linspace(150000, Q[Set].shape[0], num = 5)), axis = 0) if Set == 'p' else np.linspace(51, Q[Set].shape[0], num = 10)
        L = np.linspace(500, M, num = 50)

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

            with open(path + 'uncertainty_' + Set + '.obj', 'wb') as f:
                pickle.dump([L, V, MV, ERR, MERR], f)           

if 0:  
    L = np.linspace(500, M, num = 50) 
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
        


def fix(L, V, MV, ERR, MERR):
    for i in range(1, len(V)):
        for j in range(2): 
            V[i][j] = V[i-1][j] if V[i][j] > 1 else V[i][j]

    for i in range(1, len(MV)):
        MV[i] = MV[i-1] if MV[i] > 1 else MV[i]

    for i in range(1, len(ERR)):
        ERR[i] = ERR[i-1] if ERR[i] > 10 else ERR[i]

    for i in range(1, len(MERR)):
        MERR[i] = MERR[i-1] if MERR[i] > 0.8 else MERR[i]


    L = L[1:-1]
    V = V[1:-1]
    MV = MV[1:-1]
    ERR = ERR[1:-1]
    MERR = MERR[1:-1]

    return L, V, MV, ERR, MERR

plt.figure(1, figsize=(12,7))

if 1:
    Set = sets[0]
    with open(path + 'uncertainty_' + Set + '.obj', 'rb') as f:
        L, V, MV, ERR, MERR = pickle.load(f)

    L, V, MV, ERR, MERR = fix(L, V, MV, ERR, MERR)
    
    ax = plt.subplot(2,2,1)
    plt.plot(L, V, '.-', label = 'planned')

    ax = plt.subplot(2,2,2)
    plt.plot(L, MV, '.-', label = 'planned')

    ax = plt.subplot(2,2,3)
    plt.plot(L, ERR, '.-', label = 'planned')

    ax = plt.subplot(2,2,4)
    plt.plot(L, MERR, '.-', label = 'planned')
# except:
#     pass

try:
    Set = sets[2]
    with open(path + 'uncertainty_' + Set + '.obj', 'rb') as f:
        L, V, MV, ERR, MERR = pickle.load(f)

    L, V, MV, ERR, MERR = fix(L, V, MV, ERR, MERR)

    ax = plt.subplot(2,2,1)
    plt.plot(L, V, '.-', label = Set + ' steps')

    ax = plt.subplot(2,2,2)
    plt.plot(L, MV, '.-', label = Set + ' steps')

    ax = plt.subplot(2,2,3)
    plt.plot(L, ERR, '.-', label = Set + ' steps')

    ax = plt.subplot(2,2,4)
    plt.plot(L, MERR, '.-', label = Set + ' steps')
except:
    pass

try:
    Set = sets[1]
    with open(path + 'uncertainty_' + Set + '.obj', 'rb') as f:
        L, V, MV, ERR, MERR = pickle.load(f)

    L, V, MV, ERR, MERR = fix(L, V, MV, ERR, MERR)
    
    ax = plt.subplot(2,2,1)
    plt.plot(L, V, '.-', label = Set + ' steps')

    ax = plt.subplot(2,2,2)
    plt.plot(L, MV, '.-', label = Set + ' steps')

    ax = plt.subplot(2,2,3)
    plt.plot(L, ERR, '.-', label = Set + ' steps')

    ax = plt.subplot(2,2,4)
    plt.plot(L, MERR, '.-', label = Set + ' steps')
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

# plt.savefig(path + 'prop')


plt.figure(2, figsize=(12,7))

Set = sets[0]
with open(path + 'uncertainty_' + Set + '.obj', 'rb') as f:
    Lp, Vp, MVp, ERRp, MERRp = pickle.load(f)
Lp, Vp, MVp, ERRp, MERRp = fix(Lp, Vp, MVp, ERRp, MERRp)

Set = sets[1]
with open(path + 'uncertainty_' + Set + '.obj', 'rb') as f:
    L50, V50, MV50, ERR50, MERR50 = pickle.load(f)
L50, V50, MV50, ERR50, MERR50 = fix(L50, V50, MV50, ERR50, MERR50)

Set = sets[2]
with open(path + 'uncertainty_' + Set + '.obj', 'rb') as f:
    L10, V10, MV10, ERR10, MERR10 = pickle.load(f)
L10, V10, MV10, ERR10, MERR10 = fix( L10, V10, MV10, ERR10, MERR10)

V50, V10, Vp, ERR50, ERR10, ERRp, MV50, MV10, MVp, MERR50, MERR10, MERRp = np.array(V50), np.array(V10), np.array(Vp), np.array(ERR50), np.array(ERR10), np.array(ERRp), np.array(MV50), np.array(MV10), np.array(MVp), np.array(MERR50), np.array(MERR10), np.array(MERRp)

V50 = (V50-Vp)/Vp
V10 = (V10-Vp)/Vp
MV50 = (MV50-MVp)/MVp
MV10 = (MV10-MVp)/MVp
ERR50 = (ERR50-ERRp)/ERRp
ERR10 = (ERR10-ERRp)/ERRp
MERR50 = (MERR50-MERRp)/MERRp
MERR10 = (MERR10-MERRp)/MERRp

ax = plt.subplot(2,2,1)
plt.plot(Lp, V10)
plt.plot(Lp, V50)
plt.title('Variance')
plt.legend()

ax = plt.subplot(2,2,2)
plt.plot(Lp, V10)
plt.plot(Lp, V50)
plt.title('Max. variance')
plt.legend()

ax = plt.subplot(2,2,3)
plt.plot(Lp, ERR10)
plt.plot(Lp, ERR50)
plt.title('MSE')
plt.legend()

ax = plt.subplot(2,2,4)
plt.plot(Lp, MERR10)
plt.plot(Lp, MERR50)
plt.title('Max. error')
plt.legend()

print ERR10
print ERR50

plt.show()
