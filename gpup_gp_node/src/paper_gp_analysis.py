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
tr = '1'
stepSize = var.stepSize_

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/acrobot_test/discrete/'
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

# from gp import GaussianProcess
# from data_load import data_load
# DL = data_load(simORreal = 'acrobot', discreteORcont = 'discrete', K = 100)

# for j in range(500):

#     # j = 424

#     s = Smean[j,:]
#     a = np.array([A[j]])
#     sa = np.concatenate((s, a), axis=0)
#     sa = DL.normz( sa )    
#     st = time.time()
#     # idx = DL.kdt.query(np.copy(sa).reshape(1,-1), k = DL.K)[0]
#     # print "NN Time: ", time.time() - st
#     idx = DL.kdt.kneighbors(np.copy(sa).reshape(1,-1), n_neighbors = DL.K, return_distance=False)
#     X_nn = DL.Xtrain[idx,:].reshape(DL.K, DL.state_action_dim)
#     Y_nn = DL.Ytrain[idx,:].reshape(DL.K, DL.state_dim)
#     # print np.sort(idx)
#     ds_next = np.zeros((DL.state_dim,))
#     std_next_normz = np.zeros((DL.state_dim,))
#     Theta = DL.get_theta(sa)
#     # print Theta
#     for i in range(DL.state_dim):
#         gp_est = GaussianProcess(X_nn[:,:DL.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i])
#         mm, vv = gp_est.predict(sa[:DL.state_action_dim])
#         # print vv
#         ds_next[i] = mm
#         std_next_normz[i] = np.sqrt(vv)
#     sa_normz = sa[:DL.state_dim] + ds_next
#     s_next = DL.denormz( sa_normz )
#     std_next1 = DL.denormz_change( std_next_normz )

#     # print s_next, std_next1

#     S = Smean[j,:].reshape(1,-1)
#     a = np.array([A[j]]).reshape(1,1)

#     SA = np.concatenate((S, a), axis=1)
#     SA = DL.normz_batch( SA )
#     sa = np.mean(SA, 0)
#     st = time.time()
#     # idx = DL.kdt.query(np.copy(sa).reshape(1,-1), k = DL.K)[0]
#     # print "NN Time: ", time.time() - st
#     idx = DL.kdt.kneighbors(np.copy(sa).reshape(1,-1), n_neighbors = DL.K, return_distance=False)
#     X_nn = DL.Xtrain[idx,:].reshape(DL.K, DL.state_action_dim)
#     Y_nn = DL.Ytrain[idx,:].reshape(DL.K, DL.state_dim)
#     # print np.sort(idx)
#     dS_next = np.zeros((SA.shape[0], DL.state_dim))
#     std_next_normz = np.zeros((SA.shape[0], DL.state_dim))
#     Theta = DL.get_theta(sa)
#     # print Theta
#     for i in range(DL.state_dim):
#         gp_est = GaussianProcess(X_nn[:,:DL.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i])
#         mm, vv = gp_est.batch_predict(SA[:,:DL.state_action_dim])
#         # print vv
#         dS_next[:,i] = mm
#         std_next_normz[:,i] = np.sqrt(np.diag(vv))
#     SA_normz = SA[:,:DL.state_dim] + dS_next

#     S_next = DL.denormz_batch( SA_normz )
#     std_next2 = np.zeros(std_next_normz.shape)
#     for i in range(std_next_normz.shape[0]):
#         std_next2[i] = DL.denormz_change(std_next_normz[i])

#     # print S_next, std_next2
    
#     print j, "-------"

#     if np.any(np.abs(std_next1 - std_next2[0]) > 1e-5):
#         print std_next1 - std_next2[0]
#         print s_next, std_next1
#         print S_next, std_next2
#         raw_input()


# exit(1)

# s = np.copy(s_start)
# S = np.tile(s, (1,1))# + np.random.normal(0, sigma_start, (Np, state_dim))

# print("Running (open loop) path...")
# for i in range(0, A.shape[0]):
#     print("Step " + str(i) + " of " + str(A.shape[0]) + ", action: " + str(A[i]))
#     a = np.array([A[i]])

#     res = gp_srv(S.reshape(-1,1), a)
#     S_next = np.array(res.next_states).reshape(-1,state_dim)
#     s_mean_next = np.mean(S_next, 0)
#     s_std_next = np.std(S_next, 0)
#     S = S_next

#     res = naive_srv(s.reshape(-1,1), a)
#     s_next = np.array(res.next_state)
#     s = s_next

#     print S, s

# exit(1)

if 1:   
    Np = 1000 # Number of particles

    ######################################## GP propagation ##################################################

    print "Running GP."
    
    t_gp = 0

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
        t_gp += (time.time() - st) 

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

    t_gp /= A.shape[0]

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

    ######################################## Mean propagation ##################################################

    print "Running Batch Mean."
    Np = 100 # Number of particles

    t_mean = 0

    s = np.copy(s_start)
    S = np.tile(s, (Np,1))
    Ypred_bmean = s.reshape(1,state_dim)

    print("Running (open loop) path...")
    p_mean = 1.
    for i in range(0, A.shape[0]):
        print("[Mean] Step " + str(i) + " of " + str(A.shape[0]))
        a = np.array([A[i]])

        # st = time.time()
        # res = gp_srv(S.reshape(-1,1), a)
        # t_mean += (time.time() - st) 

        # if res.node_probability < p_mean:
        #     p_mean = res.node_probability
        # S_next = np.array(res.next_states).reshape(-1,state_dim)
        # s_mean_next = np.mean(S_next, 0)
        # S = np.tile(s_mean_next, (Np,1))

        s_mean_next = np.ones((1,state_dim))

        Ypred_bmean = np.append(Ypred_bmean, s_mean_next.reshape(1,state_dim), axis=0)

    t_mean /= A.shape[0]

    ######################################## GPUP propagation ###############################################

    print "Running GPUP."
    sigma_start += np.ones((state_dim,))*1e-4

    t_gpup = 0

    s = np.copy(s_start)
    sigma_x = sigma_start
    Ypred_mean_gpup = s.reshape(1,state_dim)
    Ypred_std_gpup = sigma_x.reshape(1,state_dim)

    print("Running (open loop) path...")
    p_gpup = 1
    for i in range(0, A.shape[0]):
        print("[GPUP] Step " + str(i) + " of " + str(A.shape[0]))
        a = np.array([A[i]])

        # st = time.time()
        # res = gpup_srv(s, sigma_x, a)
        # t_gpup += (time.time() - st) 

        # if res.node_probability < p_gpup:
        #     p_gpup = res.node_probability
        # s_next = np.array(res.next_mean)
        # sigma_next = np.array(res.next_std)
        # s = s_next
        # sigma_x = sigma_next

        s_next = sigma_next = np.ones((1,state_dim))

        Ypred_mean_gpup = np.append(Ypred_mean_gpup, s_next.reshape(1,state_dim), axis=0) #Ypred_mean_gpup,np.array([0,0,0,0]).reshape(1,state_dim),axis=0)#
        Ypred_std_gpup = np.append(Ypred_std_gpup, sigma_next.reshape(1,state_dim), axis=0)

    t_gpup /= A.shape[0]

    ######################################## Save ###########################################################

    stats = np.array([[t_gp, t_naive, t_mean, t_gpup,], [p_gp, p_naive, p_mean, p_gpup]])

    with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A], f)

######################################## Plot ###########################################################

with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A = pickle.load(f)  


t = range(A.shape[0]+1)
t = list(np.array(t)*stepSize)

plt.figure(1)
for i in range(1,5):
    ax = plt.subplot(2,2,i)

    ax.plot(range(Smean.shape[0]), Smean[:,i-1], '-b', label='rollout mean')
    # ax.fill_between(t[:-1], Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[0]]-Sstd[:,ix[0]], facecolor='blue', alpha=0.5, label='rollout std.')
    ax.plot(t, Ypred_mean_gp[:,i-1], '-r', label='BPP mean')
    ax.fill_between(t, Ypred_mean_gp[:,i-1]+Ypred_std_gp[:,i-1], Ypred_mean_gp[:,i-1]-Ypred_std_gp[:,i-1], facecolor='green', alpha=0.5, label='BGP std.')
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
plt.plot(Ypred_naive[:,ix[0]], Ypred_naive[:,ix[1]], '.-k', label='Naive')
# plt.plot(Ypred_bmean[:,ix[0]], Ypred_bmean[:,ix[1]], '.-k', label='Mean')
plt.legend()

# ax2 = plt.subplot(1,2,2)
# ix = [2, 3]
# plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-b', label='rollout mean')
# plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
# plt.plot(Ypred_naive[:,ix[0]], Ypred_naive[:,ix[1]], '.--k', label='Naive')
# plt.plot(Ypred_bmean[:,ix[0]], Ypred_bmean[:,ix[1]], '.-k', label='Mean')

plt.savefig(path + 'path_' + tr + '.png', dpi=300)
plt.show()


