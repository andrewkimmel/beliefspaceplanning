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

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)
# inter_srv = rospy.ServiceProxy('/inter/transitionOneParticle', one_transition)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
plot_srv = rospy.ServiceProxy('/rollout/plot', Empty)

##########################################################################################################
if tr == '3':
    # Rollout 1
    A = np.concatenate( (np.array([[1., 1.] for _ in range(int(3*1./stepSize))]), 
            np.array([[-1.,  1.] for _ in range(int(2*1./stepSize))]), 
            np.array([[ 1.,  0.] for _ in range(int(4*1./stepSize))]), 
            np.array([[ 1., -1.] for _ in range(int(5*1./stepSize))]),
            np.array([[-1.,  1.] for _ in range(int(6*1./stepSize))]) ), axis=0 )
if tr == '2':
    # Rollout 2
    A = np.concatenate( (np.array([[ 1., -1.] for _ in range(int(100*1./stepSize))]), 
            np.array([[-1., -1.] for _ in range(int(40*1./stepSize))]), 
            np.array([[-1.,  1.] for _ in range(int(80*1./stepSize))]),
            np.array([[ 1.,  0.] for _ in range(int(70*1./stepSize))]),
            np.array([[ 1., -1.] for _ in range(int(70*1./stepSize))]) ), axis=0 )
if tr == '1':
    A = np.array([[1., 1.] for _ in range(int(20*1./stepSize))]) 

######################################## Roll-out ##################################################


rospy.init_node('verification_gazebo', anonymous=True)

path = '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

if 1:
    Af = A.reshape((-1,))
    Pro = []
    for j in range(10):
        print("Rollout number " + str(j) + ".")
        
        R = rollout_srv(Af)
        Sro = np.array(R.states).reshape(-1,state_dim)
        # A = np.array(R.actions_res).reshape(-1,2)

        Pro.append(Sro)
        
        with open(path + 'ver_toy_rollout_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f: 
            pickle.dump(Pro, f)

f = path + 'ver_toy_rollout_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize)
with open(f + '.pkl') as f:  
    Pro = pickle.load(f) 


# fig = plt.figure(0)
# ax = fig.add_subplot(111)#, aspect='equal')
S = []
c = 0
for j in range(len(Pro)): 
    Sro = Pro[j]
    # ax.plot(Sro[:,0], Sro[:,1], 'b')
    plt.plot(Sro[:,0], Sro[:,1], '.-r')
    S.append(Sro[0,:state_dim])
    print Sro.shape[0]
    if Sro.shape[0]>=A.shape[0]:
        c+= 1
s_start = np.mean(np.array(S), 0)
sigma_start = np.std(np.array(S), 0) + np.concatenate((np.array([0.,0.]), np.ones((state_dim-2,))*1e-3), axis=0)
# ax.plot(s_start_mean[0], s_start_mean[1], 'om')
# patch = Ellipse(xy=(s_start[0], s_start[1]), width=sigma_start[0]*2, height=sigma_start[1]*2, angle=0., animated=False, edgecolor='r', linewidth=2., linestyle='-', fill=True)
# ax.add_artist(patch)

# from scipy.io import savemat
# savemat(path + 'test_v12_d4_m1.mat', {'S': Pro[0], 'A': A })
# exit(1)

Smean = []
Sstd = []
for i in range(A.shape[0]):
    F = []
    for j in range(len(Pro)): 
        if Pro[j].shape[0] > i:
            F.append(Pro[j][i])
    Smean.append( np.mean(np.array(F), axis=0) )
    Sstd.append( np.std(np.array(F), axis=0) )
Smean = np.array(Smean)
Sstd = np.array(Sstd)

print("Roll-out success rate: " + str(float(c) / len(Pro)*100) + "%")

Smean = Pro[0]

# plt.show()
# exit(1)

if 1:   
    Np = 100 # Number of particles

    ######################################## GP propagation ##################################################

    print "Running GP."
    
    t_gp = 0

    s = np.copy(s_start)
    S = np.tile(s, (Np,1))# + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_mean_gp = s.reshape(1,state_dim)
    Ypred_std_gp = sigma_start.reshape(1,state_dim)

    Pgp = []; 
    p_gp = 1
    print("Running (open loop) path...")
    for i in range(0, A.shape[0]):
        print("[GP] Step " + str(i) + " of " + str(A.shape[0]))
        Pgp.append(S)
        a = A[i,:]

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

    # t_gp /= A.shape[0]
    

    ######################################## naive propagation ###############################################

    print "Running Naive."
    Np = 1 # Number of particles
    t_naive = 0

    s = np.copy(s_start) + np.random.normal(0, sigma_start)
    # s = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_naive = s.reshape(1,state_dim)

    print("Running (open loop) path...")
    p_naive = 1
    for i in range(0, A.shape[0]):
        print("[Naive] Step " + str(i) + " of " + str(A.shape[0]))
        a = A[i,:]

        st = time.time()
        res = naive_srv(s.reshape(-1,1), a)
        t_naive += (time.time() - st) 

        if res.node_probability < p_naive:
            p_naive = res.node_probability
        s_next = np.array(res.next_state)
        s = s_next

        # s_next = np.ones((1,state_dim))

        Ypred_naive = np.append(Ypred_naive, s_next.reshape(1,state_dim), axis=0)

    # t_naive /= A.shape[0]

    ######################################## Mean propagation ##################################################

    print "Running Batch Mean."
    Np = 100 # Number of particles

    t_mean = 0

    s = np.copy(s_start)
    S = np.tile(s, (Np,1))
    Ypred_bmean = s.reshape(1,state_dim)

    print("Running (open loop) path...")
    p_mean = 1
    for i in range(0, A.shape[0]):
        print("[Mean] Step " + str(i) + " of " + str(A.shape[0]))
        a = A[i,:]

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

    # t_mean /= A.shape[0]

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
        a = A[i,:]

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

    # t_gpup /= A.shape[0]

    ######################################## Interpolation propagation ###############################################

    print "Running Interpolation."
    Np = 1 # Number of particles
    t_inter = 0

    s = np.copy(s_start) + np.random.normal(0, sigma_start)
    # s = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_inter = s.reshape(1,state_dim)

    print("Running (open loop) path...")
    p_inter = 1
    for i in range(0, A.shape[0]):
        print("[inter] Step " + str(i) + " of " + str(A.shape[0]))
        a = A[i,:]

        # st = time.time()
        # res = inter_srv(s.reshape(-1,1), a)
        # t_inter += (time.time() - st) 

        # if res.node_probability < p_inter:
        #     p_inter = res.node_probability
        # s_next = np.array(res.next_state)
        # s = s_next

        s_next = np.ones((1,state_dim))

        Ypred_inter = np.append(Ypred_inter, s_next.reshape(1,state_dim), axis=0)

    # t_inter /= A.shape[0]

    ######################################## Save ###########################################################

    stats = np.array([[t_gp, t_naive, t_mean, t_gpup,], [p_gp, p_naive, p_mean, p_gpup]])

    with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_inter, Ypred_bmean, stats, A], f)

######################################## Plot ###########################################################


with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_inter, Ypred_bmean, stats, A = pickle.load(f)  

# Compare paths
# d_gp = d_gpup = d_naive = d_mean = d = 0.
# for i in range(A.shape[0]):
#     if i < Smean.shape[0]-1:
#         d += np.linalg.norm(Smean[i,:2]-Smean[i+1,:2])
#     d_gp += np.linalg.norm(Ypred_mean_gp[i,:2] - Smean[i,:2])
#     d_naive += np.linalg.norm(Ypred_bmean[i,:2] - Smean[i,:2])
#     d_mean += np.linalg.norm(Ypred_naive[i,:2] - Smean[i,:2])
#     d_gpup += np.linalg.norm(Ypred_mean_gpup[i,:2] - Smean[i,:2])
# d_gp = np.sqrt(d_gp/A.shape[0])
# d_naive = np.sqrt(d_naive/A.shape[0])
# d_mean = np.sqrt(d_mean/A.shape[0])
# d_gpup = np.sqrt(d_gpup/A.shape[0])

# print "-----------------------------------"
# print "Path length: " + str(d)
# print "-----------------------------------"
# print "GP rmse: " + str(d_gp) + "mm"
# print "Naive rmse: " + str(d_naive) + "mm"
# print "mean rmse: " + str(d_mean) + "mm"
# print "GPUP rmse: " + str(d_gpup) + "mm"
# print "-----------------------------------"
# print "GP runtime: " + str(stats[0][0]) + "sec."
# print "GP Naive: " + str(stats[0][2]) + "sec."
# print "GP mean: " + str(stats[0][1]) + "sec."
# print "GPUP time: " + str(stats[0][3]) + "sec."
# print "-----------------------------------"
# print "GP probability: " + str(stats[1][0])
# print "GP naive probability: " + str(stats[1][1])
# print "GP mean probability: " + str(stats[1][2])
# print "GPUP probability: " + str(stats[1][3])
# print "-----------------------------------"



# plt.figure(2)
plt.plot(Smean[:,0], Smean[:,1], '.--b', label='rollout mean')
plt.plot(Ypred_mean_gp[:,0], Ypred_mean_gp[:,1], '.-g', label='BPP mean')
plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '.-k', label='Naive')
# plt.plot(Ypred_inter[:,0], Ypred_inter[:,1], '.-y', label='Inter')
plt.legend()


# if tr == '1':

#     plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-b', label='rollout mean')
#     # X = np.concatenate((Smean[:,ix[0]]+Sstd[:,ix[0]], np.flip(Smean[:,ix[0]]-Sstd[:,ix[0]])), axis=0)
#     # Y = np.concatenate((Smean[:,ix[1]]+Sstd[:,ix[1]], np.flip(Smean[:,ix[1]]-Sstd[:,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'b')

#     plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
#     X = np.concatenate((Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], np.flip(Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]])), axis=0)
#     Y = np.concatenate((Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], np.flip(Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'r')

#     # plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
#     # X = np.concatenate((Ypred_mean_gpup[:,ix[0]]+Ypred_std_gpup[:,ix[0]], np.flip(Ypred_mean_gpup[:,ix[0]]-Ypred_std_gpup[:,ix[0]])), axis=0)
#     # Y = np.concatenate((Ypred_mean_gpup[:,ix[1]]+Ypred_std_gpup[:,ix[1]], np.flip(Ypred_mean_gpup[:,ix[1]]-Ypred_std_gpup[:,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'c')

#     plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '.-k', label='Naive')
#     # plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')
#     plt.legend()

# if tr == '2':
#     ax1 = plt.subplot(1,2,1)
#     plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '-b', label='rollout mean')
#     # t = 16
#     # X = np.concatenate((Smean[:t,ix[0]]+Sstd[:t,ix[0]], np.flip(Smean[:t,ix[0]]-Sstd[:t,ix[0]])), axis=0)
#     # Y = np.concatenate((Smean[:t,ix[1]]+Sstd[:t,ix[1]], np.flip(Smean[:t,ix[1]]-Sstd[:t,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'b')
#     # t1 = t+10
#     # X = np.concatenate((Smean[t:t1,ix[0]]+Sstd[t:t1,ix[0]], np.flip(Smean[t:t1,ix[0]]-Sstd[t:t1,ix[0]])), axis=0)
#     # Y = np.concatenate((Smean[t:t1,ix[1]]+Sstd[t:t1,ix[1]], np.flip(Smean[t:t1,ix[1]]-Sstd[t:t1,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'b')
#     # t2 = Smean.shape[0]
#     # X = np.concatenate((Smean[t1-1:t2,ix[0]]+Sstd[t1-1:t2,ix[0]], np.flip(Smean[t1-1:t2,ix[0]]+Sstd[t1-1:t2,ix[0]])), axis=0)
#     # Y = np.concatenate((Smean[t1-1:t2,ix[1]]+Sstd[t1-1:t2,ix[1]], np.flip(Smean[t1-1:t2,ix[1]]-Sstd[t1-1:t2,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'b')
#     # # plt.plot(Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[1]]+Sstd[:,ix[1]], '--b', label='rollout mean')
#     # # plt.plot(Smean[:,ix[0]]-Sstd[:,ix[0]], Smean[:,ix[1]]-Sstd[:,ix[1]], '--b', label='rollout mean')

#     # plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '-r', label='BPP mean')
#     # t = 15
#     # X = np.concatenate((Ypred_mean_gp[:t,ix[0]]+Ypred_std_gp[:t,ix[0]], np.flip(Ypred_mean_gp[:t,ix[0]]-Ypred_std_gp[:t,ix[0]])), axis=0)
#     # Y = np.concatenate((Ypred_mean_gp[:t,ix[1]]+Ypred_std_gp[:t,ix[1]], np.flip(Ypred_mean_gp[:t,ix[1]]-Ypred_std_gp[:t,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'r')
#     # t1 = t+9
#     # X = np.concatenate((Ypred_mean_gp[t:t1,ix[0]]+Ypred_std_gp[t:t1,ix[0]], np.flip(Ypred_mean_gp[t:t1,ix[0]]-Ypred_std_gp[t:t1,ix[0]])), axis=0)
#     # Y = np.concatenate((Ypred_mean_gp[t:t1,ix[1]]+Ypred_std_gp[t:t1,ix[1]], np.flip(Ypred_mean_gp[t:t1,ix[1]]-Ypred_std_gp[t:t1,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'r')
#     # t2 = Ypred_mean_gp.shape[0]
#     # X = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[0]]+Ypred_std_gp[t1-1:t2,ix[0]], np.flip(Ypred_mean_gp[t1-1:t2,ix[0]]-Ypred_std_gp[t1-1:t2,ix[0]])), axis=0)
#     # Y = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[1]]+Ypred_std_gp[t1-1:t2,ix[1]], np.flip(Ypred_mean_gp[t1-1:t2,ix[1]]-Ypred_std_gp[t1-1:t2,ix[1]])), axis=0)
#     # plt.fill( X, Y , alpha = 0.5 , color = 'r')
#     # # plt.plot(Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')
#     # # plt.plot(Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')

#     # # plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
#     # # X = np.concatenate((Ypred_mean_gpup[:,ix[0]]+Ypred_std_gpup[:,ix[0]], np.flip(Ypred_mean_gpup[:,ix[0]]-Ypred_std_gpup[:,ix[0]])), axis=0)
#     # # Y = np.concatenate((Ypred_mean_gpup[:,ix[1]]+Ypred_std_gpup[:,ix[1]], np.flip(Ypred_mean_gpup[:,ix[1]]-Ypred_std_gpup[:,ix[1]])), axis=0)
#     # # plt.fill( X, Y , alpha = 0.5 , color = 'c')

#     plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '-k', label='Naive')
#     # plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')

#     ax2 = plt.subplot(1,2,2)
#     plt.plot(Smean[:,ix[0]+2], Smean[:,ix[1]+2], '-b', label='rollout mean')
#     plt.plot(Ypred_naive[:,2], Ypred_naive[:,3], '--k', label='Naive')

# if tr == '35':

#     plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '-b', label='rollout mean')
#     t = 27
#     X = np.concatenate((Smean[:t,ix[0]]+Sstd[:t,ix[0]], np.flip(Smean[:t,ix[0]]-Sstd[:t,ix[0]])), axis=0)
#     Y = np.concatenate((Smean[:t,ix[1]]+Sstd[:t,ix[1]], np.flip(Smean[:t,ix[1]]-Sstd[:t,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'b')
#     t1 = t+17
#     X = np.concatenate((Smean[t:t1,ix[0]]+Sstd[t:t1,ix[0]], np.flip(Smean[t:t1,ix[0]]-Sstd[t:t1,ix[0]])), axis=0)
#     Y = np.concatenate((Smean[t:t1,ix[1]]+Sstd[t:t1,ix[1]], np.flip(Smean[t:t1,ix[1]]-Sstd[t:t1,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'b')
#     t2 = Smean.shape[0]
#     X = np.concatenate((Smean[t1-1:t2,ix[0]]+Sstd[t1-1:t2,ix[0]], np.flip(Smean[t1-1:t2,ix[0]]-Sstd[t1-1:t2,ix[0]])), axis=0)
#     Y = np.concatenate((Smean[t1-1:t2,ix[1]]+Sstd[t1-1:t2,ix[1]], np.flip(Smean[t1-1:t2,ix[1]]-Sstd[t1-1:t2,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'b')
#     # plt.plot(Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[1]]+Sstd[:,ix[1]], '--b', label='rollout mean')
#     # plt.plot(Smean[:,ix[0]]-Sstd[:,ix[0]], Smean[:,ix[1]]-Sstd[:,ix[1]], '--b', label='rollout mean')

#     plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '-r', label='BPP mean')
#     t = 26
#     X = np.concatenate((Ypred_mean_gp[:t,ix[0]]+Ypred_std_gp[:t,ix[0]], np.flip(Ypred_mean_gp[:t,ix[0]]-Ypred_std_gp[:t,ix[0]])), axis=0)
#     Y = np.concatenate((Ypred_mean_gp[:t,ix[1]]+Ypred_std_gp[:t,ix[1]], np.flip(Ypred_mean_gp[:t,ix[1]]-Ypred_std_gp[:t,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'r')
#     t1 = t+17
#     X = np.concatenate((Ypred_mean_gp[t:t1,ix[0]]+Ypred_std_gp[t:t1,ix[0]], np.flip(Ypred_mean_gp[t:t1,ix[0]]-Ypred_std_gp[t:t1,ix[0]])), axis=0)
#     Y = np.concatenate((Ypred_mean_gp[t:t1,ix[1]]+Ypred_std_gp[t:t1,ix[1]], np.flip(Ypred_mean_gp[t:t1,ix[1]]-Ypred_std_gp[t:t1,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'r')
#     t2 = Ypred_mean_gp.shape[0]
#     X = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[0]]+Ypred_std_gp[t1-1:t2,ix[0]], np.flip(Ypred_mean_gp[t1-1:t2,ix[0]]-Ypred_std_gp[t1-1:t2,ix[0]])), axis=0)
#     Y = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[1]]+Ypred_std_gp[t1-1:t2,ix[1]], np.flip(Ypred_mean_gp[t1-1:t2,ix[1]]-Ypred_std_gp[t1-1:t2,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'r')
#     # plt.plot(Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')
#     # plt.plot(Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')

#     plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
#     t = 26
#     X = np.concatenate((Ypred_mean_gpup[:t,ix[0]]+Ypred_std_gpup[:t,ix[0]], np.flip(Ypred_mean_gpup[:t,ix[0]]-Ypred_std_gp[:t,ix[0]])), axis=0)
#     Y = np.concatenate((Ypred_mean_gpup[:t,ix[1]]+Ypred_std_gpup[:t,ix[1]], np.flip(Ypred_mean_gpup[:t,ix[1]]-Ypred_std_gp[:t,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'c')
#     t1 = t+17
#     X = np.concatenate((Ypred_mean_gpup[t:t1,ix[0]]+Ypred_std_gpup[t:t1,ix[0]], np.flip(Ypred_mean_gpup[t:t1,ix[0]]-Ypred_std_gpup[t:t1,ix[0]])), axis=0)
#     Y = np.concatenate((Ypred_mean_gpup[t:t1,ix[1]]+Ypred_std_gpup[t:t1,ix[1]], np.flip(Ypred_mean_gpup[t:t1,ix[1]]-Ypred_std_gpup[t:t1,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'c')
#     t2 = Ypred_mean_gp.shape[0]
#     X = np.concatenate((Ypred_mean_gpup[t1-1:t2,ix[0]]+Ypred_std_gpup[t1-1:t2,ix[0]], np.flip(Ypred_mean_gpup[t1-1:t2,ix[0]]-Ypred_std_gpup[t1-1:t2,ix[0]])), axis=0)
#     Y = np.concatenate((Ypred_mean_gpup[t1-1:t2,ix[1]]+Ypred_std_gpup[t1-1:t2,ix[1]], np.flip(Ypred_mean_gpup[t1-1:t2,ix[1]]-Ypred_std_gpup[t1-1:t2,ix[1]])), axis=0)
#     plt.fill( X, Y , alpha = 0.5 , color = 'c')
#     # plt.plot(Ypred_mean_gpup[:,ix[0]]+Ypred_std_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]]+Ypred_std_gpup[:,ix[1]], '--c', label='rollout mean')
#     # plt.plot(Ypred_mean_gpup[:,ix[0]]-Ypred_std_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]]-Ypred_std_gpup[:,ix[1]], '--c', label='rollout mean')

#     plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '-k', label='Naive')
#     plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')

#     ix = [0, 10, 20, 30, 42, 48]
#     for i in range(len(ix)):
#         plt.plot(Pgp[ix[i]][:,0], Pgp[ix[i]][:,1], '.k')

# if tr == '3' or tr == '4' or tr == '5' or tr == '6':
#     plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '-b', label='rollout mean')
#     # plt.plot(Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[1]]+Sstd[:,ix[1]], '--b', label='rollout mean')
#     # plt.plot(Smean[:,ix[0]]-Sstd[:,ix[0]], Smean[:,ix[1]]-Sstd[:,ix[1]], '--b', label='rollout mean')

#     # plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '-r', label='BPP mean')

#     # plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
    
#     plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '-k', label='Naive')
#     # plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')

# plt.savefig('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/temp2/path' + str(np.random.randint(100000)) + '.png')
plt.show()


