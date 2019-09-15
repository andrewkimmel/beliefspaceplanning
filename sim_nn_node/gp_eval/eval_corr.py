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

o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
c_srv = rospy.ServiceProxy('/nn/critic_seq', critic_seq)
rospy.init_node('gp_eval', anonymous=True)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/'

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    return np.sqrt(Sum / S1.shape[0])

with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
    D = pickle.load(f)
del D[:432] # Delete data that was used for training

n = 5
l_prior = 40
h = l_prior*n
if 0:
    try:
        with open(path + 'eval_corr.pkl', 'rb') as f: 
            E = pickle.load(f)
    except:
        E = []
    N = 10000
    for k in range(len(E), N):
        print str(float(k)/N*100) + '%', len(E)
        ix = np.random.randint(len(D))
        jx = np.random.randint(D[ix].shape[0]-h)

        ec = 0
        for i in range(1,n):
            H = D[ix][(i-1)*l_prior:i*l_prior, 4:6]

            l = 0
            while l < 30 and np.all(D[ix][i*l_prior] == D[ix][i*l_prior+l]):
                l += 1

            S = D[ix][jx:jx+l,:4]
            A = D[ix][jx:jx+l,4:6]

            ec += c_srv(S[0].reshape(-1,1), A[0].reshape(-1,1), l, H.reshape(-1,1)).err

        S = D[ix][jx:jx+n*l_prior,:4]
        A = D[ix][jx:jx+n*l_prior,4:6]
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

        E.append(np.array([ec, ep]))

        if not k % 100:
            with open(path + 'eval_corr.pkl', 'wb') as f: 
                pickle.dump(E, f)
else:
    with open(path + 'eval_corr.pkl', 'rb') as f: 
        E = pickle.load(f)

E = np.array(E)
plt.plot(E[:,0], E[:,1], '.')
plt.xlabel('Critic error (mm)')
plt.ylabel('Model error (mm)')
plt.show()
    
