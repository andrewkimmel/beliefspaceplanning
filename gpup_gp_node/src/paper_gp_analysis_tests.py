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
stepSize = var.stepSize_

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
naive_srv = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
plot_srv = rospy.ServiceProxy('/rollout/plot', Empty)

path =  '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/images/'

# 1
# tr = 1
# f = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal92.0_82.0_n3382_plan'
# action_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal92.0_82.0_n3382_plan.txt'
# planned_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal92.0_82.0_n3382_traj.txt'

# 2
# tr = 2
# f = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal-24.0_65.0_n6371_plan'
# action_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal-24.0_65.0_n6371_plan.txt'
# planned_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal-24.0_65.0_n6371_traj.txt'

# 3
tr = 3
f = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal-25.0_65.0_n97320_plan'
action_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal-25.0_65.0_n97320_plan.txt'
planned_file = '/home/pracsys/catkin_ws/src/t42_control/hand_control/plans/robust_goal-25.0_65.0_n97320_traj.txt'


with open(f + '.pkl') as f:  
    S = pickle.load(f) 

A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]

Splan = np.loadtxt(planned_file, delimiter=',', dtype=float)

s_start = S[0,:]
sigma_start = np.ones((state_dim,))*1e-3

def medfilter(x, W):
    w = int(W/2)
    x_new = np.copy(x)
    for i in range(0, x.shape[0]):
        if i < w:
            x_new[i] = np.mean(x[:i+w])
        elif i > x.shape[0]-w:
            x_new[i] = np.mean(x[i-w:])
        else:
            x_new[i] = np.mean(x[i-w:i+w])
    return x_new

print('Smoothing data...')
for i in range(4):
    S[:,i] = medfilter(S[:,i], 20)

Smean = S

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

    with open(path + 'test' + str(tr) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A], f)

######################################## Plot ###########################################################

with open(path + 'test' + str(tr) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A = pickle.load(f)  

t = range(A.shape[0]+1)

T = ['Position x', 'Position y', 'Load 1', 'Load 2']
plt.figure(1, figsize=(20,15))
for i in range(1,5):
    ax = plt.subplot(2,2,i)

    ax.plot(t[:len(Smean[:,i-1])], Smean[:,i-1], '.-k', label='rollout mean')
    ax.plot(t[:len(Splan[:,i-1])], Splan[:,i-1], '--m', label='Planned path')
    # ax.fill_between(t[:-1], Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[0]]-Sstd[:,ix[0]], facecolor='blue', alpha=0.5, label='rollout std.')
    ax.plot(t[:len( Ypred_mean_gp[:,i-1])], Ypred_mean_gp[:,i-1], '.-r', label='BPP mean')
    ax.fill_between(t[:len( Ypred_mean_gp[:,i-1])], Ypred_mean_gp[:,i-1]+Ypred_std_gp[:,i-1], Ypred_mean_gp[:,i-1]-Ypred_std_gp[:,i-1], facecolor='red', alpha=0.5, label='BPP std.')
    # ax.plot(t, Ypred_mean_gpup[:,0], '--c', label='GPUP mean')
    # ax.fill_between(t, Ypred_mean_gpup[:,0]+Ypred_std_gpup[:,0], Ypred_mean_gpup[:,0]-Ypred_std_gpup[:,0], facecolor='cyan', alpha=0.5, label='GPUP std.')
    ax.plot(t, Ypred_naive[:,i-1], '.-b', label='Naive')
    # ax.plot(t, Ypred_bmean[:,0], '-m', label='Batch mean')
    ax.legend()
    plt.title(T[i-1])
    plt.grid()
plt.savefig(path + 'path_time_' + str(tr) + '.png', dpi=300)

plt.figure(2, figsize=(17,12))
ax1 = plt.subplot(1,2,1)
ix = [0, 1]
plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-k', label='rollout mean')
plt.plot(Splan[:,ix[0]], Splan[:,ix[1]], '--m', label='Planned path')
plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
plt.plot(Ypred_naive[:,ix[0]], Ypred_naive[:,ix[1]], '.-b', label='Naive')
# plt.plot(Ypred_bmean[:,ix[0]], Ypred_bmean[:,ix[1]], '.-k', label='Mean')
plt.legend()

ax2 = plt.subplot(1,2,2)
ix = [2, 3]
plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '.-k', label='rollout mean')
plt.plot(Splan[:,ix[0]], Splan[:,ix[1]], '--m', label='Planned path')
plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '.-r', label='BPP mean')
plt.plot(Ypred_naive[:,ix[0]], Ypred_naive[:,ix[1]], '.-b', label='Naive')
# plt.plot(Ypred_bmean[:,ix[0]], Ypred_bmean[:,ix[1]], '.-k', label='Mean')


plt.savefig(path + 'path_state_' + str(tr) + '.png', dpi=300)
plt.show()


