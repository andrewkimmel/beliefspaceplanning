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
    G = 'sakh'
    with open(path + 'data_P' + str(l_prior) + '_' + G + '_v1.pkl', 'rb') as f: 
        X, E = pickle.load(f)

    M = 100#int(0.005*X.shape[0])
    OtrainAll = X[:-M]
    Otest = X[-M:]
    EtrainAll = E[:-M]
    Etest = E[-M:]
    d = OtrainAll.shape[1]
    n = OtrainAll.shape[0]

    R = np.linspace(0.05, 1, 20)

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

    with open(path + 'eval_critic.pkl', 'wb') as f: 
        pickle.dump([R, Hmean, Hstd, n, M], f)
else:
    with open(path + 'eval_critic.pkl', 'rb') as f: 
        R, Hmean, Hstd, n, M = pickle.load(f)

plt.figure(1, figsize = (8,3.5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.fill_between(R*100, np.array(Hmean)-np.array(Hstd)/3., np.array(Hmean)+np.array(Hstd)/3.)
plt.plot(R*100, Hmean, '-k')
plt.xlabel('portion of training data (%)')
plt.ylabel('prediction error (mm)')
plt.xlim([5, 100])
plt.savefig(path + '/eval_critic.png', dpi=200)

# plt.show()


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

plt.figure(2, figsize = (8,3.5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.bar(t, e)
plt.xticks(np.arange(0, 4, 1)) 
plt.xlabel('number of action changes')
plt.ylabel('error (mm)')
plt.savefig(path + '/eval_action_changes.png', dpi=200)
plt.show()
            

        























