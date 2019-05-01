#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import gpup_transition, batch_transition, one_transition
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout_node.srv import rolloutReq
from gp_sim_node.srv import sa_bool
import time
import var

# np.random.seed(10)

state_dim = var.state_dim_
tr = '3'
stepSize = var.stepSize_

rospy.wait_for_service('/gp/transition')
gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

path = '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/acrobot_tests/discrete/'
action_file = 'acrobot_ao_rrt_plan' + tr + '.txt'
traj_file = 'acrobot_ao_rrt_traj' + tr + '.txt'

Ar = np.loadtxt(path + action_file, delimiter=',')
A = []
for v in Ar:
    a = v[0]
    n = int(v[1]*100)
    for _ in range(n):
        A.append(a)
A = np.array(A)#[:150]

Smean = np.loadtxt(path + traj_file, delimiter=',')#[:150,:]

# plt.plot(Smean[:,0], Smean[:,1], '.-r')

s_start = Smean[0]
sigma_start = np.ones((state_dim,))*1e-5

# plt.title('path ' + tr)
# plt.show()
# exit(1)

if 1:   
    Np = 100 # Number of particles

    ######################################## GP propagation ##################################################

    print "Running GP."
    
    t_gp = []

    s = np.copy(s_start)
    S = np.tile(s, (Np,1))# + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_mean_gp = s.reshape(1,state_dim)
    Ypred_std_gp = sigma_start.reshape(1,state_dim)

    Pgp = []; 
    p_gp = 1.
    print("Running (open loop) path...")
    for i in range(0, A.shape[0]):
        print("[GP] Step " + str(i) + " of " + str(A.shape[0]) + ", action: " + str(A[i]))
        Pgp.append(S)
        a = np.array([A[i]])

        st = time.time()
        res = gp_srv(S.reshape(-1,1), a)
        t_gp.append(time.time() - st) 

        S_next = np.array(res.next_states).reshape(-1,state_dim)
        if res.node_probability < p_gp:
            p_gp = res.node_probability
        s_mean_next = np.mean(S_next, 0)
        s_std_next = np.std(S_next, 0)
        S = S_next

        # s_mean_next = np.ones((1,state_dim))
        # s_std_next = np.ones((1,state_dim))

        Ypred_mean_gp = np.append(Ypred_mean_gp, s_mean_next.reshape(1,state_dim), axis=0)
        Ypred_std_gp = np.append(Ypred_std_gp, s_std_next.reshape(1,state_dim), axis=0)

    ######################################## Brute GP propagation ##################################################

    print "Running brute GP."
    
    s = np.copy(s_start)
    S = np.tile(s, (10,1))# + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_mean_bgp = s.reshape(1,state_dim)
    Ypred_std_bgp = sigma_start.reshape(1,state_dim)

    Pbgp = []; 
    print("Running (open loop) path...")
    for i in range(0, A.shape[0]):
        print("[Brute GP] Step " + str(i) + " of " + str(A.shape[0]) + ", action: " + str(A[i]))
        Pbgp.append(S)
        a = np.array([A[i]])

        # S_next = []
        # for s in S:
        #     res = naive_srv(s.reshape(-1,1), a)
        #     S_next.append(np.array(res.next_state))

        # S_next = np.array(S_next)
        # s_mean_next = np.mean(S_next, 0)
        # s_std_next = np.std(S_next, 0)
        # S = S_next

        s_mean_next = np.zeros((1,state_dim))
        s_std_next = np.zeros((1,state_dim))

        Ypred_mean_bgp = np.append(Ypred_mean_bgp, s_mean_next.reshape(1,state_dim), axis=0)
        Ypred_std_bgp = np.append(Ypred_std_bgp, s_std_next.reshape(1,state_dim), axis=0)

    ######################################## naive propagation ###############################################

    print "Running Naive."
    Np = 1 # Number of particles
    t_naive = 0

    s = np.copy(s_start)# + np.random.normal(0, sigma_start)
    Ypred_naive = s.reshape(1,state_dim)

    print("Running (open loop) path...")
    p_naive = 1.
    for i in range(0, A.shape[0]):
        print("[Naive] Step " + str(i) + " of " + str(A.shape[0]) + ", action: " + str(A[i]))
        a = np.array([A[i]])

        st = time.time()
        res = naive_srv(s.reshape(-1,1), a)
        t_naive += (time.time() - st) 

        if res.node_probability < p_naive:
            p_naive = res.node_probability
        s_next = np.array(res.next_state)
        s = s_next

        # s_next = np.ones((1,state_dim))

        Ypred_naive = np.append(Ypred_naive, s_next.reshape(1,state_dim), axis=0)

    t_naive /= A.shape[0]


    ######################################## Save ###########################################################

    Ypred_mean_gpup = Ypred_std_gpup = 0
    Ypred_bmean = 0

    stats = np.array([[t_gp, t_naive], [p_gp, p_naive]])

    with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_bgp, Ypred_std_bgp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A], f)

######################################## Plot ###########################################################

with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_bgp, Ypred_std_bgp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A = pickle.load(f)  

print "p(gp) = " + str(stats[1,0])
print "p(naive) = " + str(stats[1,1])

t = range(A.shape[0]+1)
t = list(np.array(t)*stepSize)

plt.figure(1)
for i in range(1,5):
    ax = plt.subplot(2,2,i)

    #     ax.plot(t, S[:,i-1], '.-r', label='closed loop')
    for j in range(Pgp[0].shape[0]):
        G = []
        for k in range(len(Pgp)):
            G.append(Pgp[k][j,:])
        G = np.array(G)
        ax.plot(np.array(range(G.shape[0]))*1, G[:,i-1], ':k')#, label='particle')

    ax.plot(range(Smean.shape[0]), Smean[:,i-1], '-b', label='rollout mean')
    # ax.fill_between(t[:-1], Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[0]]-Sstd[:,ix[0]], facecolor='blue', alpha=0.5, label='rollout std.')
    ax.plot(t, Ypred_mean_gp[:,i-1], '-r', label='BPP - mean')
    ax.fill_between(t, Ypred_mean_gp[:,i-1]+Ypred_std_gp[:,i-1], Ypred_mean_gp[:,i-1]-Ypred_std_gp[:,i-1], facecolor='green', alpha=0.5, label='BPP std.')
    # ax.plot(t, Ypred_mean_bgp[:,i-1], '-c', label='BPP brute - mean')
    # ax.fill_between(t, Ypred_mean_bgp[:,i-1]+Ypred_std_bgp[:,i-1], Ypred_mean_bgp[:,i-1]-Ypred_std_bgp[:,i-1], facecolor='magenta', alpha=0.5, label='BPP brute std.')
    # ax.plot(t, Ypred_mean_gpup[:,0], '--c', label='GPUP mean')
    # ax.fill_between(t, Ypred_mean_gpup[:,0]+Ypred_std_gpup[:,0], Ypred_mean_gpup[:,0]-Ypred_std_gpup[:,0], facecolor='cyan', alpha=0.5, label='GPUP std.')
    ax.plot(t, Ypred_naive[:,i-1], '-k', label='Naive')
    # ax.plot(t, Ypred_bmean[:,0], '-m', label='Batch mean')
    ax.legend()
    plt.title('Path ' + tr)


plt.figure(2)
# ax1 = plt.subplot(1,2,1)
ix = [0, 1]
plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-b', label='rollout mean')
plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
plt.plot(Ypred_mean_bgp[:,ix[0]], Ypred_mean_bgp[:,ix[1]], '.-m', label='BPP bruth mean')
plt.plot(Ypred_naive[:,ix[0]], Ypred_naive[:,ix[1]], '.-k', label='Naive')
# plt.plot(Ypred_bmean[:,ix[0]], Ypred_bmean[:,ix[1]], '.-k', label='Mean')
plt.legend()
plt.savefig(path + 'path_' + tr + '.png', dpi=300)


# ax2 = plt.subplot(1,2,2)
# ix = [2, 3]
# plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-b', label='rollout mean')
# plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
# plt.plot(Ypred_naive[:,ix[0]], Ypred_naive[:,ix[1]], '.--k', label='Naive')
# plt.plot(Ypred_bmean[:,ix[0]], Ypred_bmean[:,ix[1]], '.-k', label='Mean')

plt.figure(3)
t_gp = stats[0][0]
plt.plot(t_gp)
plt.ylabel('Time [sec]')

plt.show()


