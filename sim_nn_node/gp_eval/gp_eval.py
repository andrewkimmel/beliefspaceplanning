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

o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
rospy.init_node('gp_eval', anonymous=True)

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    return np.sqrt(Sum / S1.shape[0])

with open('/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
    D = pickle.load(f)
del D[:432] # Delete data that was used for training

l = 10
if 0:
    O = []
    E = []
    N = 1000000
    for k in range(N):
        print k, N, float(k)/N*100
        ix = np.random.randint(len(D))
        jx = np.random.randint(D[ix].shape[0]-l)

        S = D[ix][jx:jx+l,:4]
        A = D[ix][jx:jx+l,4:6]
        S_next = D[ix][jx:jx+l,6:]

        Sp = []
        state = S[0]
        Sp.append(state)
        i = 0
        for a in A:
            state = o_srv(state.reshape(-1,1), a.reshape(-1,1)).next_state
            state = np.array(state)
            Sp.append(state)
        Sp = np.array(Sp)

        e = tracking_error(S, Sp)
        O.append(D[ix][jx,:10])
        E.append(e)

    O = np.array(O)
    E = np.array(E)

    with open('error_points' + str(l) + '.pkl', 'wb') as f: 
        pickle.dump([O, E, l], f)
else:
    with open('error_points' + str(l) + '.pkl', 'rb') as f: 
        O, E, l = pickle.load(f)

# On = []
# En = []
# for o, e in zip(O, E):
#     # if e < 0.47:
#     if np.all(o[4:6] == np.array([1,-1])):
#         On.append(o)
#         En.append(e)
# O = np.array(On)
# E = np.array(En)

gridsize = 20
plt.hexbin(O[:,0], O[:,1], C=E, gridsize=gridsize, cmap=cm.jet, bins=None)
plt.colorbar()
# plt.show()
# exit(1)

M = -10
Otrain = O[:M,:6]
Otest = O[M:,:6]
Etrain = E[:M]
Etest = E[M:]

kdt = KDTree(Otrain, leaf_size=100, metric='euclidean')
K = 100
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
i = 0
for e, o in zip(Etest, Otest):
    print i
    print o
    idx = kdt.query(o[:6].reshape(1,-1), k = K, return_distance=False)
    O_nn = Otrain[idx,:].reshape(K, 6)
    E_nn = Etrain[idx].reshape(K, 1)

    gpr = GaussianProcessRegressor(kernel=kernel).fit(O_nn, E_nn)
    e_mean, e_std = gpr.predict(o.reshape(1, -1), return_std=True)
    print e, e_mean
    exit(1)

    i += 1