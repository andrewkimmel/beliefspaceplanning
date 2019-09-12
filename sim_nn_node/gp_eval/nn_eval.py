#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from gpup_gp_node.srv import one_transition
from sim_nn_node.srv import load_model, critic

# o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
# lm_srv = rospy.ServiceProxy('/nn/load_model', load_model)
# rospy.init_node('nn_eval', anonymous=True)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/'

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    return np.sqrt(Sum / S1.shape[0])

R = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
# R = [0.8, 0.9]

# L = [5, 10, 20, 30, 40, 50]
L = [10, 20, 30]


with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
    D = pickle.load(f)

if 0:
    Hmean = []
    Hstd = []
    for l in L:
        Fmean = dict()
        Fstd = dict()
        for ratio in R:
            lm_srv(ratio)

            with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_cont_v0_d4_m1_episodes.obj', 'rb') as f: 
                D = pickle.load(f)
            r = int((1-ratio)*len(D)) 
            del D[:r] # Delete data that was used for training
                
            np.random.seed(100)

            N = 1000
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

            Fmean[ratio] = np.mean(np.array(E))
            Fstd[ratio] = np.std(np.array(E))
        Hmean.append(Fmean)
        Hstd.append(Fstd)

    with open(path + 'evan_nn.pkl', 'wb') as f: 
        pickle.dump([L, Hmean, Hstd], f)
else:
    with open(path + 'evan_nn.pkl', 'r') as f: 
        [L, Hmean, Hstd] = pickle.load(f)

fig = plt.figure(figsize=(8, 3.))
plt.gcf().subplots_adjust(bottom=0.15)
M = {10: '-', 20: '--', 30: ':'}
for l, Fmean, Fstd in zip(L, Hmean, Hstd):
    K = sorted(Fmean.keys())
    Vmean = []
    Vstd = []
    for k in K:
        Vmean.append(Fmean[k])
        Vstd.append(Fstd[k])
    K = 100-np.flipud(np.array(K))
    Vmean = np.flipud(np.array(Vmean))
    Vstd = np.flipud(np.array(Vstd))
    # plt.fill_between(K, Vmean-Vstd/4, Vmean+Vstd/4)
    plt.plot(K, Vmean, M[l], color='k', label=str(l) + ' steps pred.', linewidth=2.5)   
plt.xlabel('percentage data used (%)')
plt.ylabel('error (mm)')
plt.legend(loc='upper right')
plt.xlim([0, 60])
plt.savefig(path + '/eval_nn.png', dpi=300)
plt.show()
