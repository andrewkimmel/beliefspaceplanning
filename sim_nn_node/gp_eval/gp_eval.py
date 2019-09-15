#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from gpup_gp_node.srv import one_transition
from sklearn.neighbors import KDTree

# o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
# rospy.init_node('gp_eval', anonymous=True)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/'

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    return np.sqrt(Sum / S1.shape[0])

# with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
#     D = pickle.load(f)
# del D[:432] # Delete data that was used for training

l_prior = 40
if 0:
    if 1:
        O = []
        M = []
        E = []
        Apr = []
    else:
        with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/error_points_P' + str(l_prior) + '_v1.pkl', 'rb') as f: 
            O, M, Apr, E = pickle.load(f)
            O = list(O)
            E = list(E)
            Apr = list(Apr)
            M = list(M)
    N = 1000000*2
    for k in range(len(O), N):
        ix = np.random.randint(len(D))
        l = 10
        jx = np.random.randint(D[ix].shape[0]-l)

        h = 1
        while h < l and np.all(D[ix][jx, 4:6] == D[ix][jx + h, 4:6]):
            h += 1
        if h < 10:
            continue
        l = np.minimum(h, l)

        S = D[ix][jx:jx+l,:4]
        A = D[ix][jx:jx+l,4:6]
        S_next = D[ix][jx:jx+l,6:]

        H = D[ix][np.maximum(0, jx-l_prior):jx, 4:6]
        Hl = np.copy(H)
        if H.shape[0] < l_prior:
            H = np.concatenate((np.zeros((l_prior-H.shape[0], 2)), H), axis=0) # Validate size!!!
        Sl = D[ix][np.maximum(0, jx-l_prior):jx, :4]
        
        Sp = []
        state = S[0]
        Sp.append(state)
        i = 0
        for a in A:
            state = o_srv(state.reshape(-1,1), a.reshape(-1,1)).next_state
            state = np.array(state)
            Sp.append(state)
        Sp = np.array(Sp)
        ep = tracking_error(S, Sp)

        if Sl.shape[0] > 0:
            SL = []
            state = Sl[0]
            SL.append(state)
            i = 0
            for a in Hl:
                state = o_srv(state.reshape(-1,1), a.reshape(-1,1)).next_state
                state = np.array(state)
                SL.append(state)
            SL = np.array(SL)
            el = tracking_error(Sl, SL)
        else:
            el = 0.0

        e = ep + el
        o = np.concatenate((S[0], A[0]), axis = 0)
        O.append(o)
        M.append(l)
        Apr.append(H)
        E.append(e)

        print k, A[0], l, e

        if k > 1 and not k % 2000:
            O1 = np.array(O)
            M1 = np.array(M)
            Apr1 = np.array(Apr)
            E1 = np.array(E)

            with open(path + 'error_points_P' + str(l_prior) + '_v1.pkl', 'wb') as f: 
                pickle.dump([O1, M1, Apr1, E1], f)
else:
    # pass
    with open(path + 'error_points_P' + str(l_prior) + '_v1.pkl', 'r') as f: 
        O, L, Apr, E = pickle.load(f)

#### Heat maps ####
# try:
#     with open(path + 'temp1.pkl', 'r') as f: 
#         O, E = pickle.load(f)
# except:
#     with open(path + 'error_points_P' + str(l_prior) + '_v1.pkl', 'r') as f: 
#         O, L, Apr, E = pickle.load(f)

#     On = []
#     En = []
#     for o, e in zip(O, E):
#         if e < 7. and o[4] > 0.6 and o[5] < -0.6:#   np.all(o[4:6] > 0.6):
#             On.append(o)
#             En.append(e)
#     O = np.array(On)
#     E = np.array(En)
#     with open(path + 'temp.pkl', 'w') as f: 
#         pickle.dump([O, E], f)

# gridsize = 30
# plt.hexbin(O[:,0], O[:,1], C = E, gridsize=gridsize, cmap=cm.jet, bins=None)
# plt.colorbar()
# # plt.axis('equal')
# plt.xlabel('x (mm)')
# plt.ylabel('y (mm)')
# plt.grid()
# plt.savefig(path + '/heatmap_right.png', dpi=200)
# # plt.show()
# exit(1)
#### Heat maps ####

# print O.shape
# print "Data of size %d loaded."%O.shape[0]
M = 1000


# G = 'sak' # state, action, k steps
G = 'sakh' # state, action, k steps, prior actions
# G = 'sakt' # state, action, k_steps, prior turns
# G = 'sakpca' # state, action, k_steps, pca

if G == 'sak':
    Otrain = np.concatenate((O[:M,:6], L[:M].reshape(-1,1)), axis = 1)
    Otest = np.concatenate((O[M:,:6], L[M:].reshape(-1,1)), axis = 1)
elif G == 'sakh':
    if 0:
        with open(path + 'data_P' + str(l_prior) + '_' + G + '_v1.pkl', 'rb') as f: 
            X, E = pickle.load(f)
    else:
        X = []
        for o, apr, l in zip(O, Apr, L):
            x = np.concatenate((o[:6], apr.reshape((-1))), axis = 0) #, np.array([l])
            X.append(x)
        X = np.array(X)
        with open(path + 'data_P' + str(l_prior) + '_' + G + '_v2.pkl', 'wb') as f: 
            pickle.dump([X, E], f)
    Otrain = X[:-M]
    Otest = X[-M:]
elif G == 'sakt':
    X = []
    k = 0
    for o, apr, l in zip(O, Apr, L):
        t = 0
        i = 0
        while i < l_prior and np.all(apr[i] == np.array([0,0])):
            i += 1
        if i == 40:
            continue
        ap = apr[i]
        for j in range(i+1, apr.shape[0]):
            if not np.all(ap == apr[j]):
                ap = apr[j]
                t += 1
            
        x = np.concatenate((o[:6], np.array([l]), np.array([t])), axis = 0)
        X.append(x)
        k += 1
        print k, len(X), t
    X = np.array(X)
    Otrain = X[:-M]
    Otest = X[-M:]
elif G == 'sakpca':
    with open(path + 'data_P' + str(l_prior) + '_' + 'sakh' + '.pkl', 'rb') as f: 
        X = pickle.load(f)
    P = X[:-M,7:]
    P1 = P[:, range(0, 80, 2)]
    P2 = P[:, range(1, 80, 2)]

    from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD, KernelPCA
    pca1 = PCA(n_components = 6).fit(P1)
    pca2 = PCA(n_components = 6).fit(P2)

    Y = []
    for x in X:
        w = x[7:]
        d1 = pca1.transform(w[range(0, 80, 2)].reshape(1,-1))
        d2 = pca2.transform(w[range(1, 80, 2)].reshape(1,-1))
        y = np.concatenate((x[:7], d1.reshape((-1)), d2.reshape((-1))), axis=0)
        Y.append(y)
    Y = np.array(Y)
    Otrain = Y[:-M]
    Otest = Y[-M:]

import warnings
warnings.filterwarnings("ignore")

Etrain = E[:-M]
Etest = E[-M:]
d = Otrain.shape[1]

if 1:
    kdt = KDTree(X, leaf_size=100, metric='euclidean')
    with open(path + 'kdt_P' + str(l_prior) + '_' + G + '_v2.pkl', 'wb') as f: 
        pickle.dump(kdt, f)
else:
    with open(path + 'kdt_P' + str(l_prior) + '_' + G + '_v2.pkl', 'rb') as f: 
        kdt = pickle.load(f)
exit(1)
K = 3
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
i = 0
T = []
Err = []
import time
for e, o in zip(Etest, Otest):
    # print i
    # print o
    st = time.time()
    idx = kdt.query(o[:d].reshape(1,-1), k = K, return_distance=False)
    O_nn = Otrain[idx,:].reshape(K, d)
    E_nn = Etrain[idx].reshape(K, 1)

    gpr = GaussianProcessRegressor(kernel=kernel).fit(O_nn, E_nn)
    e_mean = gpr.predict(o.reshape(1, -1), return_std=False)[0][0]
    T.append(time.time() - st)
    Err.append(np.abs(e-e_mean))
    # print e, e_mean, np.abs(e-e_mean), o[-1]
    if i >=0:
        print e, e_mean
    i += 1

print G + ":"
print "Time: " + str(np.mean(T))
print "Error: " + str(np.mean(Err))

