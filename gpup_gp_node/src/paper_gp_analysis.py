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
tr = '4'
stepSize = var.stepSize_

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
plot_srv = rospy.ServiceProxy('/rollout/plot', Empty)

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
    A = np.array([[1., -1.] for _ in range(int(400*1./stepSize))])

######################################## Roll-out ##################################################


# from data_load import data_load
# dl = data_load(Dillute=4000)
# Dtest = dl.Qtest
# A = Dtest[:,state_dim:state_dim+2]
# Pro = []
# Pro.append(Dtest[:,:state_dim])

rospy.init_node('verification_gazebo', anonymous=True)

path = '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

if 0:
    Af = A.reshape((-1,))
    Pro = []
    for j in range(2):
        print("Rollout number " + str(j) + ' with path ' + tr + "...")
        
        R = rollout_srv(Af)
        Sro = np.array(R.states).reshape(-1,state_dim)

        Pro.append(Sro)
        
        with open(path + 'ver_rollout_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f: 
            pickle.dump(Pro, f)

f = path + 'ver_rollout_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize)
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

Smean = Pro[0]

# plt.title('path ' + tr)
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

    s = np.copy(s_start) + np.random.normal(0, sigma_start)
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

    t_gpup /= A.shape[0]

    ######################################## Save ###########################################################

    stats = np.array([[t_gp, t_naive, t_mean, t_gpup,], [p_gp, p_naive, p_mean, p_gpup]])

    with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A], f)

######################################## Plot ###########################################################

with open(path + 'ver_pred_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A = pickle.load(f)  

# # Compare paths
# d_gp = d_gpup = d_naive = d_mean = d = 0.
# for i in range(A.shape[0]):
#     if i < Smean.shape[0]-1:
#         d += np.linalg.norm(Smean[i,:2]-Smean[i+1,:2])
#     d_gp += np.linalg.norm(Ypred_mean_gp[i,:2] - Smean[i,:2])
#     d_naive += np.linalg.norm(Ypred_naive[i,:2] - Smean[i,:2])
#     d_mean += np.linalg.norm(Ypred_bmean[i,:2] - Smean[i,:2])
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
# print "GP Naive: " + str(stats[0][1]) + "sec."
# print "GP mean: " + str(stats[0][2]) + "sec."
# print "GPUP time: " + str(stats[0][3]) + "sec."
# print "-----------------------------------"
# print "GP probability: " + str(stats[1][0])
# print "GP naive probability: " + str(stats[1][1])
# print "GP mean probability: " + str(stats[1][2])
# print "GPUP probability: " + str(stats[1][3])
# print "-----------------------------------"

if 0:
    fig = plt.figure(0)
    ax = fig.add_subplot(111)#, aspect='equal')
    plt.plot(Smean[:,0], Smean[:,1], '-b')

    prtc_mean_line, = ax.plot([], [], '-g')
    sm, = ax.plot([], [], 'ok', markerfacecolor='r', markersize=8)

    prtc_mean, = ax.plot([], [], '*g')

    prtc, = ax.plot([], [], '.k', markersize=1)

    # plt.xlim(np.min(Ypred_mean_gp, 0)[0]*0-5, np.max(Ypred_mean_gp, 0)[0]*1.0)
    # plt.ylim(np.min(Ypred_mean_gp, 0)[1]*0.99, np.max(Ypred_mean_gp, 0)[1]*1.01)

    def init():
        prtc_mean.set_data([], [])
        prtc_mean_line.set_data([], [])
        sm.set_data([], [])
        prtc.set_data([], [])

        return sm, prtc_mean, prtc_mean_line, prtc,

    def animate(i):

        sm.set_data(Smean[i][0], Smean[i][1])

        prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
        prtc_mean_line.set_data(Ypred_mean_gp[:i+1,0], Ypred_mean_gp[:i+1,1])

        S = Pgp[i]
        prtc.set_data(S[:,0], S[:,1])

        return sm, prtc_mean, prtc_mean_line, prtc,

    ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=300, repeat_delay=1000, blit=True)

t = range(A.shape[0]+1)

ix = [0, 1]

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
ax1 = plt.subplot(1,2,1)
for j in range(len(Pro)): 
    Sro = Pro[j]
    plt.plot(Sro[:,0], Sro[:,1], ':y')
plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-b', label='rollout mean')
plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '.-k', label='Naive')
# plt.plot(Ypred_inter[:,0], Ypred_inter[:,1], '.-y', label='Inter')
plt.legend()

ax2 = plt.subplot(1,2,2)
for j in range(len(Pro)): 
    Sro = Pro[j]
    plt.plot(Sro[:,2], Sro[:,3], ':y')
plt.plot(Smean[:,ix[0]+2], Smean[:,ix[1]+2], '-b', label='rollout mean')
plt.plot(Ypred_mean_gp[:,ix[0]+2], Ypred_mean_gp[:,ix[1]+2], '-r', label='BPP mean')
plt.plot(Ypred_naive[:,2], Ypred_naive[:,3], '--k', label='Naive')
# plt.plot(Ypred_inter[:,2], Ypred_inter[:,3], ':y', label='Inter')


# plt.savefig('/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/temp2/path' + str(np.random.randint(100000)) + '.png', dpi=300)
plt.show()


