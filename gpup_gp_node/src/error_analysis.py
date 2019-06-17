#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import gpup_transition, batch_transition, one_transition
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
# from nn_predict.srv import StateAction2State
import time

# np.random.seed(10)

version = 14
Obj = 'cyl19'
state_dim = 4

naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)
nn_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
rospy.init_node('error_analysis', anonymous=True)

# print "Waiting for service /gp/transitionOneParticle ..."
# rospy.wait_for_service('/gp/transitionOneParticle')

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/bm/'
test_path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/bm/'

def medfilter(x, W):
    w = int(W/2)
    x_new = np.copy(x)
    for i in range(0, x.shape[0]):
        if i < w:
            continue
            # x_new[i] = np.mean(x[:i+w])
        elif i > x.shape[0]-w:
            continue
            # x_new[i] = np.mean(x[i-w:])
        else:
            x_new[i] = np.mean(x[i-w:i+w])
    return x_new

def predict_GP(s_start, A):
    print "Running GP with horizon " + str(A.shape[0])

    s = np.copy(s_start)# + np.random.normal(0, sigma_start)
    S = s.reshape(1,state_dim)

    p_naive = 1
    for i in range(0, A.shape[0]):
        # print("[GP] Step " + str(i) + " of " + str(A.shape[0]))
        a = A[i,:]

        res = naive_srv(s.reshape(-1,1), a)

        s_next = np.array(res.next_state)
        s = np.copy(s_next)

        S = np.append(S, s_next.reshape(1,state_dim), axis=0)

    return S

def predict_NN(s_start, A):
        print "Running NN with horizon " + str(A.shape[0])

        s = np.copy(s_start)# + np.random.normal(0, sigma_start)
        # s = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
        S = s.reshape(1,state_dim)

        p_naive = 1
        for i in range(0, A.shape[0]):
            # print("[NN] Step " + str(i) + " of " + str(A.shape[0]))
            a = A[i,:]

            res = nn_srv(s.reshape(-1,1), a)

            s_next = np.array(res.next_state)
            s = np.copy(s_next)

            S = np.append(S, s_next.reshape(1,state_dim), axis=0)

        return S

def tracking_error(S1, S2):
    Sum = 0.
    for s1, s2 in zip(S1, S2):
        Sum += np.linalg.norm(s1[:2]-s2[:2])**2

    l = 0.
    for i in range(1,S1.shape[0]):
        l += np.linalg.norm(S1[i,:2] - S1[i-1,:2])

    return np.sqrt(Sum / S1.shape[0]), l

def plato(G, n = 100):
    lmax = G[:,1].max()
    l = np.linspace(0, lmax, n)

    H = []
    S = []
    H.append(0.0)
    S.append(0.0)
    for i in range(1,len(l)):
        inx = np.where(np.logical_and(G[:,1] >= l[i-1],  G[:,1] <= l[i]))[0]
        H.append( np.mean(G[inx,2]) )
        S.append( np.std(G[inx, 2]) )

    H = np.array(H)
    S = np.array(S)

    inx = np.where(np.isnan(H))
    H = np.delete(H, inx, 0)
    S = np.delete(S, inx, 0)
    l = np.delete(l, inx, 0)

    return l, np.array(H), np.array(S)

## GP
if 0:
    with open(test_path + 'testpaths_' + Obj + '_d_v' + str(version) + '.pkl', 'r') as f: 
        action_seq, test_paths, Obj, Suc = pickle.load(f)

    if 0:
        with open(path + 'prediction_analysis_' + Obj + '_gp.pkl', 'r') as f: 
            Ggp = pickle.load(f)
    else: 
        Ggp = []

    j = 1
    while j < 10000:
        print("Run %d for %s, number of samples %d."%(j, Obj, len(Ggp)))
        path_inx = np.random.randint(len(test_paths))
        R = test_paths[path_inx]
        h = np.random.randint(1,np.min([380,R.shape[0]-1]))
        A = action_seq[path_inx]
        R = R[:,[0,1,2,3]]

        # A = np.concatenate((A, np.tile(R[0,:], (A.shape[0], 1))), axis=1)

        # Randomly pick a section with length h
        st_inx = np.random.randint(R.shape[0]-h-1)
        R = R[st_inx:st_inx+h]
        A = A[st_inx:st_inx+h]

        s_start = R[0,:]
        R_gp = predict_GP(s_start, A)

        e, l = tracking_error(R, R_gp)

        Ggp.append(np.array([h, l, e]))
        j += 1

        if j == 10000 or j % 5 == 0:
            with open(path + 'prediction_analysis_' + Obj + '_gp.pkl', 'w') as f: 
                pickle.dump(Ggp, f)

    with open(path + 'prediction_analysis_' + Obj + '_gp.pkl', 'w') as f: 
        pickle.dump(Ggp, f)

    Ggp = np.array(Ggp)

else:
    with open(path + 'prediction_analysis_' + Obj + '_gp.pkl', 'r') as f: 
        Ggp = np.array(pickle.load(f))

if 0:
    with open(test_path + 'testpaths_' + Obj + '_d_v' + str(version) + '.pkl', 'r') as f: 
        action_seq, test_paths, Obj, Suc = pickle.load(f)

    if 1:
        with open(path + 'prediction_analysis_' + Obj + '_nn.pkl', 'r') as f: 
            Gnn = pickle.load(f)
    else: 
        Gnn = []

    j = 1
    while j < 10000:
        print("Run %d, number of samples %d."%(j, len(Gnn)))
        path_inx = np.random.randint(len(test_paths))
        R = test_paths[path_inx]
        h = np.random.randint(180,np.min([380,R.shape[0]-1]))
        A = action_seq[path_inx]
        R = R[:,[0,1,2,3]]

        # Randomly pick a section with length h
        st_inx = np.random.randint(R.shape[0]-h-1)
        R = R[st_inx:st_inx+h]
        A = A[st_inx:st_inx+h]

        s_start = R[0,:]
        R_nn = predict_NN(s_start, A)

        e, l = tracking_error(R, R_nn)

        Gnn.append(np.array([h, l, e]))
        j += 1

        if j == 10000 or j % 200 == 0:
            with open(path + 'prediction_analysis_' + Obj + '_nn.pkl', 'w') as f: 
                pickle.dump(Gnn, f)

    Gnn = np.array(Gnn)
else:
    with open(path + 'prediction_analysis_' + Obj + '_nn.pkl', 'r') as f: 
        Gnn = np.array(pickle.load(f))
 

plt.figure(figsize=(12, 3.))
# plt.yscale('log',basey=10) 

lgp, Egp, Sgp = plato(Ggp, 50)
lnn, Enn, Snn = plato(Gnn, 50)

Egp = medfilter(Egp, 10)
Egp[:6] = medfilter(Egp[:6], 5)
Enn = medfilter(Enn, 15)

# plt.figure(figsize=(10,4))

# plt.fill_between(lgp, Egp+Sgp, Egp-Sgp, facecolor='cyan', alpha=0.5, label='GP std.')
# plt.fill_between(lnn, Enn+Snn, Enn-Snn, facecolor='red', alpha=0.5, label='NN std.')

# plt.plot(Gnn[:,1], Gnn[:,2], '.y', label = 'NN raw')
plt.plot(lnn, Enn, '-k', label = 'Neural-network')

# plt.plot(Ggp[:,1], Ggp[:,2], '.m', label = 'GP raw')
plt.plot(lgp, Egp, '-b', label = 'Gaussian Process')

plt.xlabel('Horizon (mm)', fontsize=16)
plt.ylabel('RMSE (mm)', fontsize=16)
# plt.title('Prediction error')
plt.legend()
plt.xlim([0,100])
plt.ylim([0,5])
plt.gcf().subplots_adjust(bottom=0.18)
plt.savefig(path + 'pred_sim_' + Obj + '.png', dpi=300) #str(np.random.randint(100000))
plt.show()

