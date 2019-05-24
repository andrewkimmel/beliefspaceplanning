#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import gpup_transition, batch_transition, one_transition, batch_transition_repeat
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout_node.srv import rolloutReq
from gp_sim_node.srv import sa_bool
from control.srv import pathTrackReq
import time
import var

# np.random.seed(10)

state_dim = var.state_dim_
tr = '3'
stepSize = var.stepSize_

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
gpr_srv = rospy.ServiceProxy('/gp/transitionRepeat', batch_transition_repeat)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
plot_srv = rospy.ServiceProxy('/rollout/plot', Empty)

rospy.init_node('verification_gazebo', anonymous=True)

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'



# s = np.array([-10.29469490051269531250,116.89755249023437500000,16.32597923278808593750,17.65195846557617187500,-0.01324925106018781662,0.00528503581881523132,-0.00003745680442079902,-0.00012167862587375566])
# a = np.array([-1.,-1.])

# S = np.tile(s, (100,1)) + np.random.normal(0, 0.2, (100, state_dim))

# res = gp_srv(S.reshape(-1,1), a)
# Snext = np.array(res.next_states).reshape(-1,state_dim)
# smean = np.mean(Snext, 0)
# # print 
# # print np.linalg.norm(s[:2] - np.array(res.next_states)[:2]), np.linalg.norm(np.array([-12,118]) - smean[:2])
# # print 
# # print res.mean_shift
# # print
# print res.node_probability
# print
# print res.collision_probability

# if res.collision_probability < 1.0:
#     fig, ax = plt.subplots(figsize=(12,12))
#     obs = plt.Circle(np.array([-12,118]), 2.7)#, zorder=10)
#     ax.add_artist(obs)

#     plt.plot(s[0], s[1], 'xr')
#     plt.plot(smean[0], smean[1], 'xb')

#     plt.plot(S[:,0], S[:,1], '.m')
#     plt.plot(Snext[:,0], Snext[:,1], '.k')
#     plt.axis('equal')
#     plt.show()  



# exit(1)


TR = [tr]#['1','2','3'] #
for tr in TR:

    #####################################################################################################

    if tr == '1':
        A = np.array([[-1.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,0.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [-1.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,0.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [0.00000000000000000000,1.00000000000000000000],
    [-1.00000000000000000000,-1.00000000000000000000]]) 

    if tr == '3':
        A = np.concatenate( (np.array([[-1., -1.] for _ in range(int(150*1./stepSize))]), 
                np.array([[-1.,  1.] for _ in range(int(100*1./stepSize))]), 
                np.array([[ 1.,  0.] for _ in range(int(100*1./stepSize))]), 
                np.array([[ 1., -1.] for _ in range(int(70*1./stepSize))]),
                np.array([[-1.,  1.] for _ in range(int(70*1./stepSize))]) ), axis=0 )

    if tr == '2':
        A = np.concatenate( (np.array([[ 1., -1.] for _ in range(int(100*1./stepSize))]), 
                np.array([[-1., -1.] for _ in range(int(40*1./stepSize))]), 
                np.array([[-1.,  1.] for _ in range(int(80*1./stepSize))]),
                np.array([[ 1.,  0.] for _ in range(int(70*1./stepSize))]),
                np.array([[ 1., -1.] for _ in range(int(70*1./stepSize))]) ), axis=0 )

    if tr == '4':
        A = np.array([[1., -1.] for _ in range(int(250*1./stepSize))])

    ######################################## Roll-out ##################################################

    # Open loop
    if 0:
        Af = A.reshape((-1,))
        Pro = []
        for j in range(1):
            print("Rollout number " + str(j) + ' with path ' + tr + "...")
            
            R = rollout_srv(Af)
            Sro = np.array(R.states).reshape(-1,state_dim)

            Pro.append(Sro)
            
            with open(path + 'ver_rollout_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f: 
                pickle.dump(Pro, f)

    f = path + 'ver_rollout_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize)
    print f
    with open(f + '.pkl') as f:  
        Pro = pickle.load(f) 


    # fig = plt.figure(0)
    # ax = fig.add_subplot(111)#, aspect='equal')
    S = []
    c = 0
    for j in range(len(Pro)): 
        Sro = Pro[j]
        # ax.plot(Sro[:,0], Sro[:,1], 'b')
        # plt.plot(Sro[:,0], Sro[:,1], '.-r')
        S.append(Sro[0,:state_dim])
        if Sro.shape[0]>=A.shape[0]:
            c+= 1
    s_start = np.mean(np.array(S), 0)
    sigma_start = np.std(np.array(S), 0) + np.concatenate((np.array([0.,0.]), np.ones((state_dim-2,))*1e-3), axis=0)
    # ax.plot(s_start_mean[0], s_start_mean[1], 'om')
    # patch = Ellipse(xy=(s_start[0], s_start[1]), width=sigma_start[0]*2, height=sigma_start[1]*2, angle=0., animated=False, edgecolor='r', linewidth=2., linestyle='-', fill=True)
    # ax.add_artist(patch)

    Smean = []
    Sstd = []
    for i in range(A.shape[0]+1):
        F = []
        for j in range(len(Pro)): 
            if Pro[j].shape[0] > i:
                F.append(Pro[j][i])
        Smean.append( np.mean(np.array(F), axis=0) )
        Sstd.append( np.std(np.array(F), axis=0) )
    Smean = np.array(Smean)
    Sstd = np.array(Sstd)

    # Smean = Pro[0]

    # plt.title('path ' + tr)
    # plt.show()
    # continue
    # exit(1)

    # from gp import GaussianProcess
    # from data_load import data_load
    # DL = data_load(simORreal = 'sim', discreteORcont = 'discrete', K = 100)
    # j = 40

    # s = np.copy(Smean[j,:])
    # a = A[j,:]
    # sa = np.concatenate((s, a), axis=0)
    # sa = DL.normz( sa ) 

    # idx = DL.kdt.query(np.copy(sa).reshape(1,-1), k = DL.K, return_distance=False)
    # X_nn = DL.Xtrain[idx,:].reshape(DL.K, DL.state_action_dim)
    # Y_nn = DL.Ytrain[idx,:].reshape(DL.K, DL.state_dim)
    # ds_next = np.zeros((DL.state_dim,))
    # std_next_normz = np.zeros((DL.state_dim,))
    # Theta, _ = DL.get_theta(sa)
    # for i in range(DL.state_dim):
    #     gp_est = GaussianProcess(X_nn[:,:DL.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i])
    #     mm, vv = gp_est.predict(sa[:DL.state_action_dim])
    #     ds_next[i] = mm
    #     std_next_normz[i] = np.sqrt(vv)
    # sa_normz = sa[:DL.state_dim] + ds_next
    # s_next = DL.denormz( sa_normz )
    # std_next = DL.denormz_change( std_next_normz )

    # print s_next, std_next
    # print "-----"

    # S = np.tile(Smean[j,:], (1000,1))#.reshape(1,-1)
    # a = np.tile(A[j,:], (1000,1))
    # SA = np.concatenate((S, a), axis=1)#.reshape(2,-1)
    # SA = DL.normz_batch( SA )
    # sa = np.mean(SA, 0)
    
    # idx = DL.kdt.query(np.copy(sa).reshape(1,-1), k = DL.K, return_distance=False)
    # X_nn = DL.Xtrain[idx,:].reshape(DL.K, DL.state_action_dim)
    # Y_nn = DL.Ytrain[idx,:].reshape(DL.K, DL.state_dim)

    # dS_next = np.zeros((SA.shape[0], DL.state_dim))
    # std_next_normz = np.zeros((SA.shape[0], DL.state_dim))
    # Theta, _ = DL.get_theta(sa)
    # for i in range(DL.state_dim):
    #     gp_est = GaussianProcess(X_nn[:,:DL.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i])
    #     mm, vv = gp_est.batch_predict(SA[:,:DL.state_action_dim])
    #     dS_next[:,i] = mm
    #     std_next_normz[:,i] = np.sqrt(np.diag(vv))
    # SA_normz = SA[:,:DL.state_dim] + np.random.normal(dS_next, std_next_normz)
    # S_next = DL.denormz_batch( SA_normz )
    # std_next = np.zeros(std_next_normz.shape)
    # for i in range(std_next_normz.shape[0]):
    #     std_next[i] = DL.denormz_change(std_next_normz[i])
    
    # # print S_next, std_next
    # print np.mean(S_next, 0)

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
        p_gp = 1.
        print("Running (open loop) path...")
        for i in range(0, A.shape[0]):
            print("[GP] Step " + str(i) + " of " + str(A.shape[0]))
            Pgp.append(S)
            a = A[i,:]

            st = time.time()
            res = gpr_srv(S.reshape(-1,1), a, 10, 0.65)
            t_gp += (time.time() - st) 

            S_next = np.array(res.next_states).reshape(-1,state_dim)
            if res.node_probability < p_gp:
                p_gp = res.node_probability
            s_mean_next = np.mean(S_next, 0)
            s_std_next = np.std(S_next, 0)
            S = np.copy(S_next)

            # s_mean_next = np.ones((1,state_dim))
            # s_std_next = np.ones((1,state_dim))

            Ypred_mean_gp = np.append(Ypred_mean_gp, s_mean_next.reshape(1,state_dim), axis=0)
            Ypred_std_gp = np.append(Ypred_std_gp, s_std_next.reshape(1,state_dim), axis=0)

        t_gp /= A.shape[0]

        ######################################## Brute GP propagation ##################################################

        print "Running brute GP."
        
        s = np.copy(s_start)
        S = np.tile(s, (Np,1))# + np.random.normal(0, sigma_start, (Np, state_dim))
        Ypred_mean_bgp = s.reshape(1,state_dim)
        Ypred_std_bgp = sigma_start.reshape(1,state_dim)

        Pbgp = []; 
        p_gp = 1.
        print("Running (open loop) path...")
        for i in range(0, A.shape[0]):
            print("[Bruth GP] Step " + str(i) + " of " + str(A.shape[0]) + ", action: " + str(A[i]))
            Pbgp.append(S)
            a = A[i,:]

            # S_next = []
            # for s in S:
            #     res = naive_srv(s.reshape(-1,1), a)
            #     S_next.append(np.array(res.next_state))

            # S_next = np.array(S_next)
            # if res.node_probability < p_gp:
            #     p_gp = res.node_probability
            # s_mean_next = np.mean(S_next, 0)
            # s_std_next = np.std(S_next, 0)
            # S = np.copy(S_next)

            s_mean_next = np.zeros((1,state_dim))
            s_std_next = np.zeros((1,state_dim))

            Ypred_mean_bgp = np.append(Ypred_mean_bgp, s_mean_next.reshape(1,state_dim), axis=0)
            Ypred_std_bgp = np.append(Ypred_std_bgp, s_std_next.reshape(1,state_dim), axis=0)

        # with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl') as f:  
        #     _, _, Ypred_mean_bgp, Ypred_std_bgp, _, _, _, _, _, _, _ = pickle.load(f)  


        ######################################## naive propagation ###############################################

        print "Running Naive."
        Np = 1 # Number of particles
        t_naive = 0

        s = np.copy(s_start)# + np.random.normal(0, sigma_start)
        Ypred_naive = s.reshape(1,state_dim)

        print("Running (open loop) path...")
        p_naive = 1.
        for i in range(0, A.shape[0]):
            print("[Naive] Step " + str(i) + " of " + str(A.shape[0]))
            a = A[i,:]

            st = time.time()
            res = naive_srv(s.reshape(-1,1), a)
            t_naive += (time.time() - st) 

            if res.node_probability < p_naive:
                p_naive = res.node_probability
            s_next = np.array(res.next_state)
            s = np.copy(s_next)

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

    # Closed loop
    if 0:
        Cro = []
        track_srv = rospy.ServiceProxy('/control', pathTrackReq)
        S = Ypred_mean_bgp.reshape((-1,))
        Pro = []
        for j in range(10):
            print("Rollout closed loop number " + str(j) + ' with path ' + tr + "...")
            
            R = track_srv(S.reshape((-1,)))
            Sreal = np.array(R.real_path).reshape(-1, Ypred_mean_gp.shape[1])

            Cro.append(Sreal)
            
            with open(path + 'ver_rollout_cl_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f: 
                pickle.dump(Cro, f)

    f = path + 'ver_rollout_cl_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize)
    with open(f + '.pkl') as f:  
        Cro = pickle.load(f) 



if 0:
    fig = plt.figure(0)
    ax = fig.add_subplot(111)#, aspect='equal')
    plt.plot(Smean[:,0], Smean[:,1], '-b')

    prtc_mean_line, = ax.plot([], [], '-g')
    sm, = ax.plot([], [], 'ok', markerfacecolor='r', markersize=8)

    prtc_mean, = ax.plot([], [], '*g')
    naivep, = ax.plot([],[], 'xc')
    naivep_line, = ax.plot([],[], '-c')
    brute_line, = ax.plot([],[], '-y')
    brute, = ax.plot([],[], 'oy')

    prtc, = ax.plot([], [], '.k', markersize=1)

    # plt.xlim(np.min(Ypred_mean_gp, 0)[0]*0-5, np.max(Ypred_mean_gp, 0)[0]*1.0)
    # plt.ylim(np.min(Ypred_mean_gp, 0)[1]*0.99, np.max(Ypred_mean_gp, 0)[1]*1.01)

    def init():
        prtc_mean.set_data([], [])
        prtc_mean_line.set_data([], [])
        sm.set_data([], [])
        prtc.set_data([], [])
        naivep.set_data([],[])
        naivep_line.set_data([],[])
        brute.set_data([],[])
        brute_line.set_data([],[])

        return sm, prtc_mean, prtc_mean_line, prtc, naivep, naivep_line, brute_line, brute,

    def animate(i):

        sm.set_data(Smean[i][0], Smean[i][1])

        prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
        prtc_mean_line.set_data(Ypred_mean_gp[:i+1,0], Ypred_mean_gp[:i+1,1])

        S = Pgp[i]
        prtc.set_data(S[:,0], S[:,1])

        naivep.set_data(Ypred_naive[i,0],Ypred_naive[i,1])
        naivep_line.set_data(Ypred_naive[:i+1,0], Ypred_naive[:i+1,1])

        F = np.random.normal(Ypred_mean_bgp[i+1,:], Ypred_std_bgp[i+1,:], (100, 8))
        brute.set_data(F[:,0], F[:,1])

        brute_line.set_data(Ypred_mean_bgp[:i+1,0], Ypred_mean_bgp[:i+1,1])

        return sm, prtc_mean, prtc_mean_line, prtc, naivep, naivep_line, brute_line, brute,

    ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=300, repeat_delay=1000, blit=True)
    plt.show()
    exit(1)


def align_curves(Sref, S):

    from sklearn.neighbors import NearestNeighbors

    Snew = []
    Snew.append(S[0])
    n = 50
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(S.reshape(-1,1))
    jdx_prev = jdx = 0
    for i in range(1,Sref.shape[0]):
        _, idx =nbrs.kneighbors(Sref[i])

        mn = 1e9
        for j in idx[0]:
            if np.abs(S[j]-Sref[i]) < mn and np.abs(j-jdx_prev) < 500:# j >= jdx_prev:# and np.abs(j-jdx_prev) < 150:
                jdx = j
                mn = np.abs(S[j]-Sref[i])
        jdx_prev = jdx
    0
    0
    0
    return np.array(Snew)

CLro = []
for S in Cro:
    Snew = np.zeros(Ypred_mean_bgp.shape)
    for i in range(Ypred_mean_bgp.shape[1]):
        Snew[:,i] = align_curves(Ypred_mean_bgp[:,i], S[:,i])
    CLro.append(Snew)
CLmean = []
CLstd = []
for i in range(A.shape[0]+1):
    F = []
    for j in range(len(CLro)): 
        if CLro[j].shape[0] > i:
            F.append(CLro[j][i])
    CLmean.append( np.mean(np.array(F), axis=0) )
    CLstd.append( np.std(np.array(F), axis=0) )
CLmean = np.array(CLmean)
CLstd = np.array(CLstd)
         
freq = 0.5
t = np.array(range(A.shape[0]+1))*freq
n = len(t)
plt.figure(1)
for i in range(1,3):
    ax = plt.subplot(1,2,i)

    # for S in CLro:
    #     ax.plot(t, S[:,i-1], '.-r', label='closed loop')
    # for j in range(Pgp[0].shape[0]):
    #     G = []
    #     for k in range(len(Pgp)):
    #         G.append(Pgp[k][j,:])
    #     G = np.array(G)
    #     ax.plot(np.array(range(G.shape[0]))*freq, G[:,i-1], '-k')#, label='particle')

    ax.plot(np.array(range(n))*freq, Smean[:n,i-1], '-b', label='rollout mean')
    ax.fill_between(t[:n], Smean[:n,i-1]+Sstd[:n,i-1], Smean[:n,i-1]-Sstd[:n,i-1], facecolor='blue', alpha=0.5, label='rollout std.')
    ax.plot(t, Ypred_mean_gp[:,i-1], '-r', label='BPP mean')
    ax.fill_between(t, Ypred_mean_gp[:,i-1]+Ypred_std_gp[:,i-1], Ypred_mean_gp[:,i-1]-Ypred_std_gp[:,i-1], facecolor='red', alpha=0.5, label='BPP std.')
    # ax.plot(t[:n], Ypred_mean_bgp[:n,i-1], '-c', label='BPPb - mean')
    # ax.fill_between(t[:n], Ypred_mean_bgp[:n,i-1]+Ypred_std_bgp[:n,i-1], Ypred_mean_bgp[:n,i-1]-Ypred_std_bgp[:n,i-1], facecolor='magenta', alpha=0.5, label='BPPb std.')
    # ax.plot(t, Ypred_mean_gpup[:,0], '--c', label='GPUP mean')
    # ax.fill_between(t, Ypred_mean_gpup[:,0]+Ypred_std_gpup[:,0], Ypred_mean_gpup[:,0]-Ypred_std_gpup[:,0], facecolor='cyan', alpha=0.5, label='GPUP std.')
    ax.plot(t[:n], Ypred_naive[:n,i-1], '.-k', label='Naive')
    # ax.plot(t, Ypred_bmean[:,0], '-m', label='Batch mean')
    # ax.plot(t[:n], CLmean[:n,i-1], '-r', label='CL mean')
    # ax.fill_between(t[:n], CLmean[:n,i-1]+CLstd[:n,i-1], CLmean[:n,i-1]-CLstd[:n,i-1], facecolor='red', alpha=0.5, label='CL std.')

    ax.legend()
plt.title('Path ' + tr)

ix = [0,1]

# For Juntao
if 0:
    # gp_path = Ypred_naive[:,:4]
    gp_path = Ypred_mean_gp[:,:4]
    rollout_path = Smean[:,:4]
    action_path = A
    with open(path + 'jt_path' + tr + '_v' + str(var.data_version_) + '_m' + str(stepSize) + '.pkl', 'w') as f: 
        pickle.dump([rollout_path, gp_path, action_path], f)


plt.figure(2)
# ax1 = plt.subplot(1,2,1)
for Sro in Pro: 
    plt.plot(Sro[:,0], Sro[:,1], ':y')
# for s in Cro:
#     plt.plot(s[:,0], s[:,1], '--c')
plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-b', label='rollout mean')
plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '.-k', label='Naive')
# plt.plot(Ypred_inter[:,0], Ypred_inter[:,1], '.-y', label='Inter')
plt.legend()

# # ax2 = plt.subplot(1,2,2)
# # for Sro in Pro: 
# #     plt.plot(Sro[:,2], Sro[:,3], ':y')
# # for s in Cro:
# #     plt.plot(s[:,2], s[:,3], '--c')
# # plt.plot(Smean[:,ix[0]+2], Smean[:,ix[1]+2], '-b', label='rollout mean')
# # plt.plot(Ypred_mean_gp[:,ix[0]+2], Ypred_mean_gp[:,ix[1]+2], '-r', label='BPP mean')
# # plt.plot(Ypred_naive[:,2], Ypred_naive[:,3], '--k', label='Naive')
# # # plt.plot(Ypred_inter[:,2], Ypred_inter[:,3], ':y', label='Inter')

# plt.figure(3)
# ec = [np.linalg.norm(c[:2]-g[:2]) for g, c in zip(Ypred_mean_bgp, CLmean)]
# el = [np.linalg.norm(c[:2]-g[:2]) for g, c in zip(Ypred_mean_bgp, Smean)]
# plt.plot(t[:n], ec[:n], label='closed-loop')
# plt.plot(t[:n], el[:n], label='open-loop')
# plt.legend()
# plt.xlabel('time (sec)')
# plt.ylabel('error (mm)')

# plt.savefig('/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/temp2/path' + str(np.random.randint(100000)) + '.png', dpi=300)
plt.show()


