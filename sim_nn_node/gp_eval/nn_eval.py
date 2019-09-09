#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from gpup_gp_node.srv import one_transition
from sim_nn_node.srv import load_model, critic

o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
lm_srv = rospy.ServiceProxy('/nn/load_model', load_model)
rospy.init_node('nn_eval', anonymous=True)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/'

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    return np.sqrt(Sum / S1.shape[0])

R = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# R = [0.8, 0.9]

L = [5, 10, 20, 30, 40, 50]

with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
    D = pickle.load(f)

if 1:
    H = []
    for l in L:
        F = dict()
        for ratio in R:
            lm_srv(ratio)

            with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
                D = pickle.load(f)
            r = int((1-ratio)*len(D)) 
            del D[:r] # Delete data that was used for training
                
            np.random.seed(100)

            N = 2000
            E = []
            for k in range(N):
                print l, ratio, k
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
                E.append(e)

            F[ratio] = np.mean(np.array(E))
        H.append(F)

    with open(path + 'evan_nn.pkl', 'wb') as f: 
        pickle.dump([L, H], f)
else:
    with open(path + 'evan_nn.pkl', 'r') as f: 
        [L, H] = pickle.load(f)

for l, F in zip(L, H):
    K = sorted(F.keys())
    V = []
    for k in K:
        V.append(F[k])
    K = 1-np.flipud(np.array(K))
    V = np.flipud(np.array(V))
    plt.plot(K, V, label=str(l))
plt.legend()
plt.show()
