#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import gpup_transition, batch_transition, one_transition
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import pickle
from rollout_node.srv import rolloutReq
from gp_sim_node.srv import sa_bool
import time

# np.random.seed(10)

state_dim = 4+2
tr = '2'
stepSize = 10

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
plot_srv = rospy.ServiceProxy('/rollout/plot', Empty)

##########################################################################################################
if tr == '3':
    # Rollout 1
    A = np.concatenate( (np.array([[-1., -1.] for _ in range(int(150*1./stepSize))]), 
            np.array([[-1., 1.] for _ in range(int(100*1./stepSize))]), 
            np.array([[1., 0.] for _ in range(int(100*1./stepSize))]), 
            np.array([[1., -1.] for _ in range(int(70*1./stepSize))]),
            np.array([[-1., 1.] for _ in range(int(70*1./stepSize))]) ), axis=0 )
if tr == '2':
    # Rollout 2
    A = np.concatenate( (np.array([[-1., -1.] for _ in range(int(5*1./stepSize+1))]), 
            np.array([[ 1., -1.] for _ in range(int(100*1./stepSize))]), 
            np.array([[-1., -1.] for _ in range(int(40*1./stepSize))]), 
            np.array([[-1.,  1.] for _ in range(int(80*1./stepSize))]),
            np.array([[ 1.,  0.] for _ in range(int(70*1./stepSize))]),
            np.array([[ 1., -1.] for _ in range(int(70*1./stepSize))]) ), axis=0 )
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

######################################## Roll-out ##################################################

rospy.init_node('verification_gazebo', anonymous=True)
rate = rospy.Rate(15) # 15hz

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

if 1:
    Af = A.reshape((-1,))
    Pro = []
    for j in range(100):
        print("Rollout number " + str(j) + ".")
        
        Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)

        Pro.append(Sro)

        with open(path + 'ver_rollout_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl', 'w') as f: 
            pickle.dump(Pro, f)

with open(path + 'ver_rollout_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl') as f:  
    Pro = pickle.load(f) 
# fig = plt.figure(0)
# ax = fig.add_subplot(111)#, aspect='equal')
S = []
c = 0
for j in range(len(Pro)): 
    Sro = Pro[j]
    # ax.plot(Sro[:,0], Sro[:,1], 'b')
    S.append(Sro[0,:state_dim])
    if Sro.shape[0]==Pro[0].shape[0]:
        c+= 1
s_start = np.mean(np.array(S), 0)
sigma_start = np.std(np.array(S), 0) + np.array([0.,0.,1e-4,1e-4,0,0])
# ax.plot(s_start_mean[0], s_start_mean[1], 'om')
# patch = Ellipse(xy=(s_start[0], s_start[1]), width=sigma_start[0]*2, height=sigma_start[1]*2, angle=0., animated=False, edgecolor='r', linewidth=2., linestyle='-', fill=True)
# ax.add_artist(patch)

Smean = Pro[0]

print("Roll-out success rate: " + str(float(c) / len(Pro)*100) + "%")

plt.show()
exit(1)


if 1:   

    ######################################## GP propagation ##################################################

    print "Running GP."
    Np = 500 # Number of particles
    sigma_start = np.std(np.array(S), 0) + np.array([0.,0.,1e-4,1e-4,0,0])

    t_gp = time.time()

    s = s_start
    S = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_mean_gp = s.reshape(1,state_dim)
    Ypred_std_gp = sigma_start.reshape(1,state_dim)

    Pgp = []; 
    print("Running (open loop) path...")
    for i in range(0, A.shape[0]):
        print("[GP] Step " + str(i) + " of " + str(A.shape[0]))
        Pgp.append(S)
        a = A[i,:]

        res = gp_srv(S.reshape(-1,1), a)
        S_next = np.array(res.next_states).reshape(-1,state_dim)
        s_mean_next = np.mean(S_next, 0)
        s_std_next = np.std(S_next, 0)
        S = S_next

        Ypred_mean_gp = np.append(Ypred_mean_gp, s_mean_next.reshape(1,state_dim), axis=0)
        Ypred_std_gp = np.append(Ypred_std_gp, s_std_next.reshape(1,state_dim), axis=0)

    t_gp = (time.time() - t_gp) / A.shape[0]

    ######################################## naive propagation ###############################################

    print "Running Naive."
    Np = 1 # Number of particles
    sigma_start = np.std(np.array(S), 0) + np.array([0.,0.,1e-4,1e-4,0,0])
    t_naive = time.time()

    s = s_start
    s = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_naive = s.reshape(1,state_dim)

    Pgp = []; 
    print("Running (open loop) path...")
    for i in range(0, A.shape[0]):
        print("[Naive] Step " + str(i) + " of " + str(A.shape[0]))
        Pgp.append(S)
        a = A[i,:]

        res = naive_srv(s.reshape(-1,1), a)
        s_next = np.array(res.next_state)
        s = s_next

        Ypred_naive = np.append(Ypred_naive, s_next.reshape(1,state_dim), axis=0)

    t_naive = (time.time() - t_naive) / A.shape[0]

    ######################################## Mean propagation ##################################################

    print "Running Batch Mean."
    Np = 500 # Number of particles

    t_mean = time.time()

    s = s_start
    S = np.tile(s, (Np,1))
    Ypred_bmean = s.reshape(1,state_dim)

    Pgp = []; 
    print("Running (open loop) path...")
    for i in range(0, A.shape[0]):
        print("[Mean] Step " + str(i) + " of " + str(A.shape[0]))
        Pgp.append(S)
        a = A[i,:]

        res = gp_srv(S.reshape(-1,1), a)
        S_next = np.array(res.next_states).reshape(-1,state_dim)
        s_mean_next = np.mean(S_next, 0)
        S = np.tile(s_mean_next, (Np,1))

        Ypred_bmean = np.append(Ypred_bmean, s_mean_next.reshape(1,state_dim), axis=0)

    t_mean = (time.time() - t_mean) / A.shape[0]

    ######################################## GPUP propagation ###############################################

    print "Running GPUP."
    sigma_start = np.std(np.array(S), 0) + np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4])

    t_gpup = time.time()

    s = s_start
    sigma_x = sigma_start
    Ypred_mean_gpup = s.reshape(1,state_dim)
    Ypred_std_gpup = sigma_x.reshape(1,state_dim)

    print("Running (open loop) path...")
    for i in range(0, A.shape[0]):
        print("[GPUP] Step " + str(i) + " of " + str(A.shape[0]))
        a = A[i,:]

        res = gpup_srv(s, sigma_x, a)
        s_next = np.array(res.next_mean)
        sigma_next = np.array(res.next_std)
        s = s_next
        sigma_x = sigma_next

        Ypred_mean_gpup = np.append(Ypred_mean_gpup, s_next.reshape(1,state_dim), axis=0) #Ypred_mean_gpup,np.array([0,0,0,0]).reshape(1,state_dim),axis=0)#
        Ypred_std_gpup = np.append(Ypred_std_gpup, sigma_next.reshape(1,state_dim), axis=0)

    t_gpup = (time.time() - t_gpup) / A.shape[0]

    ######################################## Save ###########################################################

    with open(path + 'ver_pred_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, t_gp, t_gpup, t_naive, t_mean, A], f)

######################################## Plot ###########################################################


with open(path + 'ver_pred_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, t_gp, t_gpup, t_naive, t_mean, A = pickle.load(f)  

print "GP runtime: " + str(t_gp) + "sec."
print "GPUP time: " + str(t_gpup) + "sec."
print "GP mean: " + str(t_mean) + "sec."
print "GPUP naive: " + str(t_naive) + "sec."

# Animate
if 1:
    fig = plt.figure(0)
    ax = fig.add_subplot(111)#, aspect='equal')
    for j in range(len(Pro)): 
        Sro = Pro[j]
        ax.plot(Sro[:,0], Sro[:,1], '.-b')

    prtc_mean_line, = ax.plot([], [], '-g')
    prtc, = ax.plot([], [], '.k', markersize=1)

    sm, = ax.plot([], [], 'ok', markerfacecolor='r', markersize=8)

    prtc_mean, = ax.plot([], [], '*g')
    patch_prtc = Ellipse(xy=(Ypred_mean_gp[0,0], Ypred_mean_gp[0,1]), width=Ypred_std_gp[0,0]*2, height=Ypred_std_gp[0,1]*2, angle=0., animated=True, edgecolor='y', linewidth=2., fill=False)
    ax.add_patch(patch_prtc)

    patch = Ellipse(xy=(Ypred_mean_gpup[0,0], Ypred_mean_gpup[0,1]), width=Ypred_std_gpup[0,0]*2, height=Ypred_std_gpup[0,1]*2, angle=0., animated=True, edgecolor='m', linewidth=2., linestyle='--', fill=False)
    ax.add_patch(patch)
    patch_mean, = ax.plot([], [], '--m')


    # plt.xlim(np.min(Ypred_mean_gp, 0)[0]*0-5, np.max(Ypred_mean_gp, 0)[0]*1.0)
    # plt.ylim(np.min(Ypred_mean_gp, 0)[1]*0.99, np.max(Ypred_mean_gp, 0)[1]*1.01)

    def init():
        prtc.set_data([], [])
        prtc_mean.set_data([], [])
        prtc_mean_line.set_data([], [])
        patch_mean.set_data([], [])
        # sm.set_data([], [])

        return prtc, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

    def animate(i):

        S = Pgp[i]
        prtc.set_data(S[:,0], S[:,1])
        # print i, len(idx_g), len(idx_f)

        # sm.set_data(Smean[i][0], Smean[i][1])

        prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
        prtc_mean_line.set_data(Ypred_mean_gp[:i+1,0], Ypred_mean_gp[:i+1,1])

        patch_prtc.center = (Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
        patch_prtc.width = Ypred_std_gp[i,0]*2
        patch_prtc.height = Ypred_std_gp[i,1]*2

        patch.center = (Ypred_mean_gpup[i,0], Ypred_mean_gpup[i,1])
        patch.width = Ypred_std_gpup[i,0]*2
        patch.height = Ypred_std_gpup[i,1]*2
        patch_mean.set_data(Ypred_mean_gpup[:i+1,0], Ypred_mean_gpup[:i+1,1])

        return prtc, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

    ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=10, repeat_delay=1000, blit=True)
    # ani.save(path + 'belief_gazebo_' + str(tr) + '_v5.mp4', metadata={'artist':'Avishai Sintov','year':'2019'}, bitrate=-1, codec="libx264")

    # plt.figure(1)
    # for k in range(4):
    #     ax1 = plt.subplot(2,2,k+1)
    #     for i in range(len(Pgp)):
    #         S = Pgp[i]
    #         ax1.plot(S[:,k], '-b')
    #     for i in range(len(Pro)):
    #         S = Pro[i]
    #         ax1.plot(S[:,k], '-k')

    plt.show()

S = np.array(Pro[0])

t = range(A.shape[0]+1)


plt.figure(1)
ax1 = plt.subplot(2,1,1)
ax1.plot(S[:,0], '.-b')
ax1.plot(Ypred_mean_gp[1:,0], '-r')
ax1.fill_between(t, Ypred_mean_gp[:,0]+Ypred_std_gp[:,0], Ypred_mean_gp[:,0]-Ypred_std_gp[:,0], facecolor='red', alpha=0.5)
ax1.plot(Ypred_mean_gpup[:,0], '--y')
ax1.fill_between(t, Ypred_mean_gpup[:,0]+Ypred_std_gpup[:,0], Ypred_mean_gpup[:,0]-Ypred_std_gpup[:,0], facecolor='yellow', alpha=0.5)
ax1.plot(Ypred_naive[:,0], '-k')
ax1.plot(Ypred_bmean[:,0], '-m')

ax2 = plt.subplot(2,1,2)
ax2.plot(S[:,1], '.-b')
ax2.plot(Ypred_mean_gp[:,1], '-r')
ax2.fill_between(t, Ypred_mean_gp[:,1]+Ypred_std_gp[:,1], Ypred_mean_gp[:,1]-Ypred_std_gp[:,1], facecolor='red', alpha=0.5)
ax2.plot(Ypred_mean_gpup[:,1], '--y')
ax2.fill_between(t, Ypred_mean_gpup[:,1]+Ypred_std_gpup[:,1], Ypred_mean_gpup[:,1]-Ypred_std_gpup[:,1], facecolor='yellow', alpha=0.5)
ax2.plot(Ypred_naive[:,1], '-k')
ax2.plot(Ypred_bmean[:,1], '-m')

plt.show()


