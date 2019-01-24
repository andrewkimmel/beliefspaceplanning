#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import gpup_transition, batch_transition
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import pickle
from rollout_node.srv import observation, IsDropped, TargetAngles
# from gp_sim_node.srv import sa_bool
import time

np.random.seed(10)

state_dim = 4+2
tr = '1'

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
# svm_srv = rospy.ServiceProxy('/svm_fail_check', sa_bool)

obs_srv = rospy.ServiceProxy('/hand_control/observation', observation)
drop_srv = rospy.ServiceProxy('/hand_control/IsObjDropped', IsDropped)
move_srv = rospy.ServiceProxy('/hand_control/MoveGripper', TargetAngles)
reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)

##########################################################################################################
if tr == '1':
    # Rollout 1
    A = np.concatenate( (np.array([[-1., -1.] for _ in range(150)]), 
            np.array([[-1., 1.] for _ in range(100)]), 
            np.array([[1., 0.] for _ in range(100)]), 
            np.array([[1., -1.] for _ in range(70)]),
            np.array([[-1., 1.] for _ in range(70)]) ), axis=0 )
if tr == '2':
    # Rollout 2
    A = np.concatenate( (np.array([[-1., -1.] for _ in range(5)]), 
            np.array([[1., -1.] for _ in range(100)]), 
            np.array([[-1., -1.] for _ in range(40)]), 
            np.array([[-1., 1.] for _ in range(80)]),
            np.array([[1., 0.] for _ in range(70)]),
            np.array([[1., -1.] for _ in range(70)]) ), axis=0 )

######################################## Roll-out ##################################################

rospy.init_node('verification_gazebo', anonymous=True)
rate = rospy.Rate(15) # 15hz

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/verf/'

if 0:
    Pro = []
    for j in range(2):
        print("Rollout number " + str(j) + ".")
        # Reset gripper
        reset_srv()

        # Start episode
        Sro = []
        for i in range(A.shape[0]):

            # Get observation and choose action
            state = np.array(obs_srv().state)
            action = A[i,:]

            Sro.append(state)
            
            suc = move_srv(action).success
            rospy.sleep(0.2) # For sim_data_discrete v5
            # rospy.sleep(0.05) # For all other
            rate.sleep()

            # Get observation
            next_state = np.array(obs_srv().state)

            if suc:
                fail = drop_srv().dropped # Check if dropped - end of episode
            else:
                # End episode if overload or angle limits reached
                rospy.logerr('[RL] Failed to move gripper. Episode declared failed.')
                fail = True

            # self.texp.add(state, action, next_state, not suc or fail)
            state = next_state

            if not suc or fail:
                print("Fail")
                Sro.append(state)
                break

        Pro.append(np.array(Sro))

        with open(path + 'ver_rollout_' + tr + '_v5.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(Pro, f)

with open(path + 'ver_rollout_' + tr + '_v5.pkl') as f:  
    Pro = pickle.load(f) 
fig = plt.figure(0)
ax = fig.add_subplot(111)#, aspect='equal')
S = []
c = 0
for j in range(len(Pro)): 
    Sro = Pro[j]
    ax.plot(Sro[:,0], Sro[:,1], 'b')
    S.append(Sro[0,:state_dim])
    if Sro.shape[0]==Pro[0].shape[0]:
        c+= 1
s_start = np.mean(np.array(S), 0)
# ax.plot(s_start_mean[0], s_start_mean[1], 'om')
# patch = Ellipse(xy=(s_start[0], s_start[1]), width=sigma_start[0]*2, height=sigma_start[1]*2, angle=0., animated=False, edgecolor='r', linewidth=2., linestyle='-', fill=True)
# ax.add_artist(patch)

Smean = []
for i in range(Sro.shape[0]):
    F = []
    for j in range(len(Pro)): 
        F.append(Pro[j][i])
    Smean.append( np.mean(np.array(F), axis=0) )

print("Roll-out success rate: " + str(float(c) / len(Pro)*100) + "%")

# plt.show()
# exit(1)

######################################## GP propagation ##################################################
Np = 500 # Number of particles
if 1:
    print "Running GP."
    sigma_start = np.std(np.array(S), 0) + np.array([0.,0.,1e-4,1e-4,0.,0.])

    t_gp = time.time()

    s = s_start
    S = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np, state_dim))
    Ypred_mean_gp = s.reshape(1,state_dim)
    Ypred_std_gp = np.zeros((1,state_dim)).reshape(1,state_dim)

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

    ######################################## GPUP propagation ###############################################

    print "Running GPUP."
    t_gpup = time.time()
    sigma_start = np.std(np.array(S), 0) + np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4])

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

    with open(path + 'ver_pred_' + tr + '_v5.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp], f)

######################################## Plot ###########################################################

with open(path + 'ver_pred_' + tr + '_v5.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp = pickle.load(f)  

if 0:
    def svm_check(S, a, idx_g, idx_f):
        idx_g_temp = []
        for j in range(len(idx_g)):
            sa = np.concatenate((S[idx_g[j],:], a), axis=0)
            res = svm_srv(sa)
            if res.probability > 0.7:
                idx_f.append(idx_g[j])
            else:
                idx_g_temp.append(idx_g[j])

        return idx_g_temp[:], idx_f[:]

    idx_g = range(0, Np)
    idx_f = []
    Ig = []
    If = []
    for i in range(len(Pgp)):
        print("SVM check, iteration " + str(i) + ".")
        S = Pgp[i]
        idx_g, idx_f = svm_check(S, A[i,:], idx_g, idx_f)
        Ig.append(idx_g)
        If.append(idx_f)
        print len(idx_g), len(idx_f)

    with open(path + 'ver_gf_' + tr + '_v3.pkl', 'w') as f:  
        pickle.dump([Ig, If], f)

with open(path + 'ver_gf_' + tr + '_v2.pkl') as f:  
    Ig, If = pickle.load(f)  

prtc_mean_line, = ax.plot([], [], '-g')
prtc_g, = ax.plot([], [], '.k', markersize=1)
prtc_f, = ax.plot([], [], '.r', markersize=1)

sm, = ax.plot([], [], 'ok', markerfacecolor='r', markersize=8)

prtc_mean, = ax.plot([], [], '*g')
patch_prtc = Ellipse(xy=(Ypred_mean_gp[0,0], Ypred_mean_gp[0,1]), width=Ypred_std_gp[0,0]*2, height=Ypred_std_gp[0,1]*2, angle=0., animated=True, edgecolor='y', linewidth=2., fill=False)
ax.add_patch(patch_prtc)

patch = Ellipse(xy=(Ypred_mean_gpup[0,0], Ypred_mean_gpup[0,1]), width=Ypred_std_gpup[0,0]*2, height=Ypred_std_gpup[0,1]*2, angle=0., animated=True, edgecolor='m', linewidth=2., linestyle='--', fill=False)
ax.add_patch(patch)
patch_mean, = ax.plot([], [], '--m')

print "GP runtime: " + str(t_gp) + "sec, " + " GPUP time: " + str(t_gpup) + "sec."

# plt.xlim(np.min(Ypred_mean_gp, 0)[0]*0-5, np.max(Ypred_mean_gp, 0)[0]*1.0)
# plt.ylim(np.min(Ypred_mean_gp, 0)[1]*0.99, np.max(Ypred_mean_gp, 0)[1]*1.01)

def init():
    prtc_g.set_data([], [])
    prtc_f.set_data([], [])
    prtc_mean.set_data([], [])
    prtc_mean_line.set_data([], [])
    patch_mean.set_data([], [])
    sm.set_data([], [])

    return sm, prtc_g, prtc_f, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

def animate(i):

    S = Pgp[i]
    idx_g = Ig[i]
    idx_f = If[i]
    prtc_g.set_data(S[idx_g,0], S[idx_g,1])
    prtc_f.set_data(S[idx_f,0], S[idx_f,1])
    # print i, len(idx_g), len(idx_f)

    sm.set_data(Smean[i][0], Smean[i][1])

    prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    prtc_mean_line.set_data(Ypred_mean_gp[:i+1,0], Ypred_mean_gp[:i+1,1])

    patch_prtc.center = (Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    patch_prtc.width = Ypred_std_gp[i,0]*2
    patch_prtc.height = Ypred_std_gp[i,1]*2

    patch.center = (Ypred_mean_gpup[i,0], Ypred_mean_gpup[i,1])
    patch.width = Ypred_std_gpup[i,0]*2
    patch.height = Ypred_std_gpup[i,1]*2
    patch_mean.set_data(Ypred_mean_gpup[:i,0], Ypred_mean_gpup[:i,1])

    return sm, prtc_g, prtc_f, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=100, repeat_delay=1000, blit=True)
ani.save(path + 'belief_gazebo_' + str(tr) + '_v5.mp4', metadata={'artist':'Avishai Sintov','year':'2019'}, bitrate=-1, codec="libx264")

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

