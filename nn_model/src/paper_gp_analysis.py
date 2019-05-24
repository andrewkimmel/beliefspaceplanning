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
from control.srv import pathTrackReq
import time

import sys
sys.path.insert(0, '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

# np.random.seed(10)

state_dim = 4
tr = '1'
stepSize = var.stepSize_

gp_srv = rospy.ServiceProxy('/nn/transition', batch_transition)
gpr_srv = rospy.ServiceProxy('/nn/transitionRepeat', batch_transition_repeat)
naive_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
plot_srv = rospy.ServiceProxy('/rollout/plot', Empty)

rospy.init_node('verification_gazebo', anonymous=True)

path = '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

TR = [tr]#['1','2','3'] #
for tr in TR:

    # with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/jt_path3_v14_m10.pkl', 'rb') as pickle_file:
    #     traj_data = pickle.load(pickle_file, encoding='latin1')
    # S = np.asarray(traj_data[0])#[:-1,:]
    # A = traj_data[2]
    with open('/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/jt_rollout_2_v14_d8_m1.pkl', 'rb') as pickle_file:
        traj_data = pickle.load(pickle_file)
    S = np.asarray(traj_data[0])[:,:4]
    A = traj_data[1]

    s_start = S[0,:]
    sigma_start = np.ones((state_dim,))*0.1
    # ax.plot(s_start_mean[0], s_start_mean[1], 'om') 
    # patch = Ellipse(xy=(s_start[0], s_start[1]), width=sigma_start[0]*2, height=sigma_start[1]*2, angle=0., animated=False, edgecolor='r', linewidth=2., linestyle='-', fill=True)
    # ax.add_artist(patch)

    Smean = S


    if 1:   
        Np = 100 # Number of particles

        ######################################## GP propagation ##################################################

        print "Running GP."
        
        t_gp = 0

        s = np.copy(s_start)
        S = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
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
            res = gp_srv(S.reshape(-1,1), a)
            t_gp += (time.time() - st) 

            S_next = np.array(res.next_states).reshape(-1,state_dim)
            if res.node_probability < p_gp:
                p_gp = res.node_probability
            s_mean_next = np.mean(S_next, 0)
            s_std_next = np.std(S_next, 0)
            S = np.copy(S_next)
            if S.shape[0] == 0:
                break

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
        Ypred_mean_bgp = Ypred_std_bgp = 0                                                                                                                                   

        stats = np.array([[t_gp, t_naive], [p_gp, p_naive]])

        with open(path + 'ver_nn_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f:
            pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_bgp, Ypred_std_bgp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A], f)

    ######################################## Plot ###########################################################

    with open(path + 'ver_nn_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl') as f:  
        Ypred_mean_gp, Ypred_std_gp, Ypred_mean_bgp, Ypred_std_bgp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A = pickle.load(f)  



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

        return sm, prtc_mean, prtc_mean_line, prtc, naivep, naivep_line

    def animate(i):

        sm.set_data(Smean[i][0], Smean[i][1])

        prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
        prtc_mean_line.set_data(Ypred_mean_gp[:i+1,0], Ypred_mean_gp[:i+1,1])

        S = Pgp[i]
        prtc.set_data(S[:,0], S[:,1])

        naivep.set_data(Ypred_naive[i,0],Ypred_naive[i,1])
        naivep_line.set_data(Ypred_naive[:i+1,0], Ypred_naive[:i+1,1])

        return sm, prtc_mean, prtc_mean_line, prtc, naivep, naivep_line

    ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=300, repeat_delay=1000, blit=True)
    plt.show()
    exit(1)





         
# freq = 0.5
# t = np.array(range(A.shape[0]+1))*freq
# n = len(t)
# plt.figure(1)
# for i in range(1,3):
#     ax = plt.subplot(1,2,i)

#     ax.plot(np.array(range(n))*freq, Smean[:n,i-1], '-b', label='rollout mean')
#     ax.plot(t, Ypred_mean_gp[:,i-1], '-r', label='BPP mean')
#     ax.fill_between(t, Ypred_mean_gp[:,i-1]+Ypred_std_gp[:,i-1], Ypred_mean_gp[:,i-1]-Ypred_std_gp[:,i-1], facecolor='red', alpha=0.5, label='BPP std.')
#     ax.plot(t[:n], Ypred_naive[:n,i-1], '.-k', label='Naive')

#     ax.legend()
# plt.title('Path ' + tr)

ix = [0,1]




plt.figure(2)
# ax1 = plt.subplot(1,2,1)
# for s in Cro:
#     plt.plot(s[:,0], s[:,1], '--c')
plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-b', label='rollout mean')
plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '.-k', label='Naive')
# plt.plot(Ypred_inter[:,0], Ypred_inter[:,1], '.-y', label='Inter')
plt.legend()


# plt.savefig('/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/temp2/path' + str(np.random.randint(100000)) + '.png', dpi=300)
plt.show()


