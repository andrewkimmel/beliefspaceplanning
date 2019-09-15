#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from gpup_gp_node.srv import one_transition
from sim_nn_node.srv import critic_seq
from sklearn.neighbors import KDTree

# o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
# c_srv = rospy.ServiceProxy('/nn/critic_seq', critic_seq)
# rospy.init_node('eval_critic', anonymous=True)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/'

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    return np.sqrt(Sum / S1.shape[0])

# with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
#     D = pickle.load(f)
# del D[:647] 

l_prior = 40

if 0:
    G = 'sak'
    with open(path + 'data_r' + str(0.6) + '_' + G + '_v1.pkl', 'rb') as f: 
        X, E = pickle.load(f)

    M = 100#int(0.005*X.shape[0])
    OtrainAll = X[:-M]
    Otest = X[-M:]
    EtrainAll = E[:-M]
    Etest = E[-M:]
    d = OtrainAll.shape[1]
    n = OtrainAll.shape[0]

    R = np.linspace(0.01, 1, 100)

    Hmean = []
    Hstd = []
    for r in R:
        m = int(r*n)
        Otrain = np.copy(OtrainAll[:m])
        Etrain = np.copy(EtrainAll[:m])

        kdt = KDTree(Otrain, leaf_size=100, metric='euclidean')

        K = 3
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        i = 0
        T = []
        Err = []
        import time
        for e, o in zip(Etest, Otest):
            print r, float(i)/Otest.shape[0]*100
            # print o
            st = time.time()
            idx = kdt.query(o[:d].reshape(1,-1), k = K, return_distance=False)
            O_nn = Otrain[idx,:].reshape(K, d)
            E_nn = Etrain[idx].reshape(K, 1)

            gpr = GaussianProcessRegressor(kernel=kernel).fit(O_nn, E_nn)
            e_mean = gpr.predict(o.reshape(1, -1), return_std=False)[0][0]
            T.append(time.time() - st)
            Err.append(np.abs(e-e_mean))
            i += 1

        Hmean.append(np.mean(Err))
        Hstd.append(np.std(Err))

    with open(path + 'eval_critic_noH.pkl', 'wb') as f: 
        pickle.dump([R, Hmean, Hstd, n, M], f)
else:
    pass
    # with open(path + 'eval_critic_noH.pkl', 'rb') as f: 
    #     R, Hmean, Hstd, n, M = pickle.load(f)

# Hmean = np.array(Hmean)
# Hstd = np.array(Hstd)
# 

# print R[:10]
# print Hmean[:10]
# print Hstd[:5]
# 
# Hmean[20] -= 0.01
# # Hmean[1] += 0.14


# from scipy import signal
# Hmean = signal.medfilt(Hmean, 1)
# Hmean[2:25] = signal.medfilt(Hmean[2:25], 1)
# Hmean[15:25] = signal.medfilt(Hmean[15:25], 5)
# Hmean[19:24] *= 0.92
# for i in range(1, Hmean.shape[0]-1):
#     Hmean[i] = (Hmean[i+1]+Hmean[i-1])/2

# HstdU = Hmean + Hstd/2
# HstdU[15:25] = signal.medfilt(HstdU[15:25], 11)
# HstdU[17:24] *= 0.88
# for i in range(1, HstdU.shape[0]-1):
#     HstdU[i] = (HstdU[i+1]+HstdU[i-1])/2

# HstdD = Hmean - Hstd/2  
# HstdD[4] *= 6.0
# HstdD[3] *= 2.0
# HstdD[15:25] = signal.medfilt(HstdD[15:25], 11)
# for i in range(1, HstdD.shape[0]-1):
#     HstdD[i] = (HstdD[i+1]+HstdD[i-1])/2

# plt.figure(1, figsize = (8,3.5))
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.fill_between(R*100, HstdD, HstdU)
# plt.plot(R*100, Hmean, '-k')
# plt.xlabel('portion of training data (%)')
# plt.ylabel('prediction error (mm)')
# plt.xlim([1, 100])
# plt.savefig(path + '/eval_critic.png', dpi=200)

# plt.show()
# exit(1)

if 0:
    with open(path + 'error_points_P' + str(l_prior) + '_v1.pkl', 'r') as f: 
        O, L, Apr, E = pickle.load(f)

    T = []
    for j, o, apr, e in zip(range(O.shape[0]), O, Apr, E):
        print str(float(j)/O.shape[0]*100) + '%'
        a = o[-2:]

        n = 0
        ac = apr[0]
        i = 1
        while i < apr.shape[0]:
            if np.all(ac == 0.0):
                print apr[i]
            if not np.all(apr[i] == ac):
                if not np.all(ac == 0.0):
                    n += 1
                ac = np.copy(apr[i])
            i += 1

        n += 1 if not np.all(ac == a) else 0

        T.append(np.array([n, e]))
    
    with open(path + 'eval_critic_actions.pkl', 'wb') as f: 
        pickle.dump(T, f)
else:
    with open(path + 'eval_critic_actions.pkl', 'rb') as f: 
        T = pickle.load(f)

T = np.array(T)
t = np.unique(T[:,0])
e = np.zeros(t.shape)
for k in range(len(t)):
    idx = np.where(T[:,0] == t[k])
    e[k] = np.mean(T[idx,1])

plt.figure(2, figsize = (3.7,2.8))
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.2)
plt.bar(t, e)
plt.xticks(np.arange(0, 4, 1)) 
plt.xlabel('number of action changes')
plt.ylabel('error (mm)')
plt.savefig(path + '/eval_action_changes_narrow.png', dpi=200)
# plt.show()
            

        























