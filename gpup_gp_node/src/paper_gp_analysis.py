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

# np.random.seed(10)

state_dim = 4+2
tr = '5'
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

if tr=='4': 
    A = np.loadtxt('/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/robust/robust_particles_pc_goal7_run3_plan.txt', delimiter=',', dtype=float)[:,:2]

if tr=='5': 
    A = np.loadtxt('/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/robust/robust_particles_pc_goal1_run1_plan.txt', delimiter=',', dtype=float)[:,:2]
      
if tr=='6': 
    A = np.loadtxt('/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/robust/robust_particles_pc_goal7_run4_plan.txt', delimiter=',', dtype=float)[:,:2]
      
######################################## Roll-out ##################################################

rospy.init_node('verification_gazebo', anonymous=True)
rate = rospy.Rate(15) # 15hz

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

if 0:
    Af = A.reshape((-1,))
    Pro = []
    for j in range(100):
        print("Rollout number " + str(j) + ".")
        
        Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)

        Pro.append(Sro)
        
        with open(path + 'ver_rollout_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl', 'w') as f: 
            pickle.dump(Pro, f)


if tr=='4':
    f = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/robust/robust_particles_pc_goal7_run3_plan'
elif tr=='5':
    f = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/robust/robust_particles_pc_goal1_run1_plan'
elif tr=='6':
    f = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/robust/robust_particles_pc_goal7_run4_plan'
else:
    f = path + 'ver_rollout_' + tr + '_v5_d6_m' + str(stepSize)
with open(f + '.pkl') as f:  
    Pro = pickle.load(f) 
# fig = plt.figure(0)
# ax = fig.add_subplot(111)#, aspect='equal')
S = []
c = 0
for j in range(len(Pro)): 
    Sro = Pro[j]
    # ax.plot(Sro[:,0], Sro[:,1], 'b')
    S.append(Sro[0,:state_dim])
    if Sro.shape[0]==A.shape[0]:
        c+= 1
s_start = np.mean(np.array(S), 0)
sigma_start = np.std(np.array(S), 0) + np.array([0.,0.,1e-4,1e-4,0,0])
# ax.plot(s_start_mean[0], s_start_mean[1], 'om')
# patch = Ellipse(xy=(s_start[0], s_start[1]), width=sigma_start[0]*2, height=sigma_start[1]*2, angle=0., animated=False, edgecolor='r', linewidth=2., linestyle='-', fill=True)
# ax.add_artist(patch)

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

# plt.show()
# exit(1)

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

        Ypred_mean_gp = np.append(Ypred_mean_gp, s_mean_next.reshape(1,state_dim), axis=0)
        Ypred_std_gp = np.append(Ypred_std_gp, s_std_next.reshape(1,state_dim), axis=0)

    t_gp /= A.shape[0]
    

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
    p_mean = 1
    for i in range(0, A.shape[0]):
        print("[Mean] Step " + str(i) + " of " + str(A.shape[0]))
        a = A[i,:]

        st = time.time()
        res = gp_srv(S.reshape(-1,1), a)
        t_mean += (time.time() - st) 

        if res.node_probability < p_mean:
            p_mean = res.node_probability
        S_next = np.array(res.next_states).reshape(-1,state_dim)
        s_mean_next = np.mean(S_next, 0)
        S = np.tile(s_mean_next, (Np,1))

        Ypred_bmean = np.append(Ypred_bmean, s_mean_next.reshape(1,state_dim), axis=0)

    t_mean /= A.shape[0]

    ######################################## GPUP propagation ###############################################

    print "Running GPUP."
    sigma_start += np.array([1e-4,1e-4,0.,0.,1e-4,1e-4])

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

        st = time.time()
        res = gpup_srv(s, sigma_x, a)
        t_gpup += (time.time() - st) 

        if res.node_probability < p_gpup:
            p_gpup = res.node_probability
        s_next = np.array(res.next_mean)
        sigma_next = np.array(res.next_std)
        s = s_next
        sigma_x = sigma_next

        Ypred_mean_gpup = np.append(Ypred_mean_gpup, s_next.reshape(1,state_dim), axis=0) #Ypred_mean_gpup,np.array([0,0,0,0]).reshape(1,state_dim),axis=0)#
        Ypred_std_gpup = np.append(Ypred_std_gpup, sigma_next.reshape(1,state_dim), axis=0)

    t_gpup /= A.shape[0]

    ######################################## Save ###########################################################

    stats = np.array([[t_gp, t_naive, t_mean, t_gpup], [p_gp, p_naive, p_mean, p_gpup]])

    with open(path + 'ver_pred_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A], f)

######################################## Plot ###########################################################


with open(path + 'ver_pred_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp, Ypred_naive, Ypred_bmean, stats, A = pickle.load(f)  

# Compare paths
d_gp = d_gpup = d_naive = d_mean = d = 0.
for i in range(A.shape[0]):
    if i < Smean.shape[0]-1:
        d += np.linalg.norm(Smean[i,:]-Smean[i+1,:])
    d_gp += np.linalg.norm(Ypred_mean_gp[i,:] - Smean[i,:])
    d_naive += np.linalg.norm(Ypred_bmean[i,:] - Smean[i,:])
    d_mean += np.linalg.norm(Ypred_naive[i,:] - Smean[i,:])
    d_gpup += np.linalg.norm(Ypred_mean_gpup[i,:] - Smean[i,:])
d_gp = np.sqrt(d_gp/A.shape[0])
d_naive = np.sqrt(d_naive/A.shape[0])
d_mean = np.sqrt(d_mean/A.shape[0])
d_gpup = np.sqrt(d_gpup/A.shape[0])

print "-----------------------------------"
print "Path length: " + str(d)
print "-----------------------------------"
print "GP rmse: " + str(d_gp) + "mm"
print "Naive rmse: " + str(d_naive) + "mm"
print "mean rmse: " + str(d_mean) + "mm"
print "GPUP rmse: " + str(d_gpup) + "mm"
print "-----------------------------------"
print "GP runtime: " + str(stats[0][0]) + "sec."
print "GP Naive: " + str(stats[0][2]) + "sec."
print "GP mean: " + str(stats[0][1]) + "sec."
print "GPUP time: " + str(stats[0][3]) + "sec."
print "-----------------------------------"
print "GP probability: " + str(stats[1][0])
print "GP naive probability: " + str(stats[1][1])
print "GP mean probability: " + str(stats[1][2])
print "GPUP probability: " + str(stats[1][3])
print "-----------------------------------"


t = range(A.shape[0]+1)

ix = [0, 1]

# plt.figure(1)
# ax1 = plt.subplot(2,1,1)
# ax1.plot(t[:-1], Smean[:,ix[0]], '-b', label='rollout mean')
# ax1.fill_between(t[:-1], Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[0]]-Sstd[:,ix[0]], facecolor='blue', alpha=0.5, label='rollout std.')
# ax1.plot(t, Ypred_mean_gp[:,ix[0]], '-r', label='BPP mean')
# ax1.fill_between(t, Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]], facecolor='red', alpha=0.5, label='BGP std.')
# ax1.plot(t, Ypred_mean_gpup[:,0], '--c', label='GPUP mean')
# ax1.fill_between(t, Ypred_mean_gpup[:,0]+Ypred_std_gpup[:,0], Ypred_mean_gpup[:,0]-Ypred_std_gpup[:,0], facecolor='cyan', alpha=0.5, label='GPUP std.')
# ax1.plot(t, Ypred_naive[:,0], '-k', label='Naive')
# ax1.plot(t, Ypred_bmean[:,0], '-m', label='Batch mean')
# ax1.legend()
# plt.title('Path ' + tr)

# ax2 = plt.subplot(2,1,2)
# ax2.plot(t[:-1], Smean[:,ix[1]], '-b')
# ax2.fill_between(t[:-1], Smean[:,ix[1]]+Sstd[:,ix[1]], Smean[:,ix[1]]-Sstd[:,ix[1]], facecolor='blue', alpha=0.5)
# ax2.plot(t, Ypred_mean_gp[:,ix[1]], '-r')
# ax2.fill_between(t, Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]], facecolor='red', alpha=0.5)
# ax2.plot(t, Ypred_mean_gpup[:,1], '--c')
# ax2.fill_between(t, Ypred_mean_gpup[:,1]+Ypred_std_gpup[:,1], Ypred_mean_gpup[:,1]-Ypred_std_gpup[:,1], facecolor='cyan', alpha=0.5)
# ax2.plot(t, Ypred_naive[:,1], '-k')
# ax2.plot(t, Ypred_bmean[:,1], '-m')

plt.figure(2)

if tr == '1':

    plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '-b', label='rollout mean')
    X = np.concatenate((Smean[:,ix[0]]+Sstd[:,ix[0]], np.flip(Smean[:,ix[0]]-Sstd[:,ix[0]])), axis=0)
    Y = np.concatenate((Smean[:,ix[1]]+Sstd[:,ix[1]], np.flip(Smean[:,ix[1]]-Sstd[:,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'b')

    plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '-r', label='BPP mean')
    X = np.concatenate((Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], np.flip(Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], np.flip(Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'r')

    plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
    X = np.concatenate((Ypred_mean_gpup[:,ix[0]]+Ypred_std_gpup[:,ix[0]], np.flip(Ypred_mean_gpup[:,ix[0]]-Ypred_std_gpup[:,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gpup[:,ix[1]]+Ypred_std_gpup[:,ix[1]], np.flip(Ypred_mean_gpup[:,ix[1]]-Ypred_std_gpup[:,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'c')

    plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '-k', label='Naive')
    plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')

if tr == '2':

    plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '-b', label='rollout mean')
    t = 16
    X = np.concatenate((Smean[:t,ix[0]]+Sstd[:t,ix[0]], np.flip(Smean[:t,ix[0]]-Sstd[:t,ix[0]])), axis=0)
    Y = np.concatenate((Smean[:t,ix[1]]+Sstd[:t,ix[1]], np.flip(Smean[:t,ix[1]]-Sstd[:t,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'b')
    t1 = t+10
    X = np.concatenate((Smean[t:t1,ix[0]]+Sstd[t:t1,ix[0]], np.flip(Smean[t:t1,ix[0]]-Sstd[t:t1,ix[0]])), axis=0)
    Y = np.concatenate((Smean[t:t1,ix[1]]+Sstd[t:t1,ix[1]], np.flip(Smean[t:t1,ix[1]]-Sstd[t:t1,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'b')
    t2 = Smean.shape[0]
    X = np.concatenate((Smean[t1-1:t2,ix[0]]+Sstd[t1-1:t2,ix[0]], np.flip(Smean[t1-1:t2,ix[0]]+Sstd[t1-1:t2,ix[0]])), axis=0)
    Y = np.concatenate((Smean[t1-1:t2,ix[1]]+Sstd[t1-1:t2,ix[1]], np.flip(Smean[t1-1:t2,ix[1]]-Sstd[t1-1:t2,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'b')
    # plt.plot(Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[1]]+Sstd[:,ix[1]], '--b', label='rollout mean')
    # plt.plot(Smean[:,ix[0]]-Sstd[:,ix[0]], Smean[:,ix[1]]-Sstd[:,ix[1]], '--b', label='rollout mean')

    plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '-r', label='BPP mean')
    t = 15
    X = np.concatenate((Ypred_mean_gp[:t,ix[0]]+Ypred_std_gp[:t,ix[0]], np.flip(Ypred_mean_gp[:t,ix[0]]-Ypred_std_gp[:t,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gp[:t,ix[1]]+Ypred_std_gp[:t,ix[1]], np.flip(Ypred_mean_gp[:t,ix[1]]-Ypred_std_gp[:t,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'r')
    t1 = t+9
    X = np.concatenate((Ypred_mean_gp[t:t1,ix[0]]+Ypred_std_gp[t:t1,ix[0]], np.flip(Ypred_mean_gp[t:t1,ix[0]]-Ypred_std_gp[t:t1,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gp[t:t1,ix[1]]+Ypred_std_gp[t:t1,ix[1]], np.flip(Ypred_mean_gp[t:t1,ix[1]]-Ypred_std_gp[t:t1,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'r')
    t2 = Ypred_mean_gp.shape[0]
    X = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[0]]+Ypred_std_gp[t1-1:t2,ix[0]], np.flip(Ypred_mean_gp[t1-1:t2,ix[0]]-Ypred_std_gp[t1-1:t2,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[1]]+Ypred_std_gp[t1-1:t2,ix[1]], np.flip(Ypred_mean_gp[t1-1:t2,ix[1]]-Ypred_std_gp[t1-1:t2,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'r')
    # plt.plot(Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')
    # plt.plot(Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')

    # plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
    # X = np.concatenate((Ypred_mean_gpup[:,ix[0]]+Ypred_std_gpup[:,ix[0]], np.flip(Ypred_mean_gpup[:,ix[0]]-Ypred_std_gpup[:,ix[0]])), axis=0)
    # Y = np.concatenate((Ypred_mean_gpup[:,ix[1]]+Ypred_std_gpup[:,ix[1]], np.flip(Ypred_mean_gpup[:,ix[1]]-Ypred_std_gpup[:,ix[1]])), axis=0)
    # plt.fill( X, Y , alpha = 0.5 , color = 'c')

    plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '-k', label='Naive')
    plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')

if tr == '3':

    plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '-b', label='rollout mean')
    t = 27
    X = np.concatenate((Smean[:t,ix[0]]+Sstd[:t,ix[0]], np.flip(Smean[:t,ix[0]]-Sstd[:t,ix[0]])), axis=0)
    Y = np.concatenate((Smean[:t,ix[1]]+Sstd[:t,ix[1]], np.flip(Smean[:t,ix[1]]-Sstd[:t,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'b')
    t1 = t+17
    X = np.concatenate((Smean[t:t1,ix[0]]+Sstd[t:t1,ix[0]], np.flip(Smean[t:t1,ix[0]]-Sstd[t:t1,ix[0]])), axis=0)
    Y = np.concatenate((Smean[t:t1,ix[1]]+Sstd[t:t1,ix[1]], np.flip(Smean[t:t1,ix[1]]-Sstd[t:t1,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'b')
    t2 = Smean.shape[0]
    X = np.concatenate((Smean[t1-1:t2,ix[0]]+Sstd[t1-1:t2,ix[0]], np.flip(Smean[t1-1:t2,ix[0]]-Sstd[t1-1:t2,ix[0]])), axis=0)
    Y = np.concatenate((Smean[t1-1:t2,ix[1]]+Sstd[t1-1:t2,ix[1]], np.flip(Smean[t1-1:t2,ix[1]]-Sstd[t1-1:t2,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'b')
    # plt.plot(Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[1]]+Sstd[:,ix[1]], '--b', label='rollout mean')
    # plt.plot(Smean[:,ix[0]]-Sstd[:,ix[0]], Smean[:,ix[1]]-Sstd[:,ix[1]], '--b', label='rollout mean')

    plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '-r', label='BPP mean')
    t = 26
    X = np.concatenate((Ypred_mean_gp[:t,ix[0]]+Ypred_std_gp[:t,ix[0]], np.flip(Ypred_mean_gp[:t,ix[0]]-Ypred_std_gp[:t,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gp[:t,ix[1]]+Ypred_std_gp[:t,ix[1]], np.flip(Ypred_mean_gp[:t,ix[1]]-Ypred_std_gp[:t,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'r')
    t1 = t+17
    X = np.concatenate((Ypred_mean_gp[t:t1,ix[0]]+Ypred_std_gp[t:t1,ix[0]], np.flip(Ypred_mean_gp[t:t1,ix[0]]-Ypred_std_gp[t:t1,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gp[t:t1,ix[1]]+Ypred_std_gp[t:t1,ix[1]], np.flip(Ypred_mean_gp[t:t1,ix[1]]-Ypred_std_gp[t:t1,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'r')
    t2 = Ypred_mean_gp.shape[0]
    X = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[0]]+Ypred_std_gp[t1-1:t2,ix[0]], np.flip(Ypred_mean_gp[t1-1:t2,ix[0]]-Ypred_std_gp[t1-1:t2,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gp[t1-1:t2,ix[1]]+Ypred_std_gp[t1-1:t2,ix[1]], np.flip(Ypred_mean_gp[t1-1:t2,ix[1]]-Ypred_std_gp[t1-1:t2,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'r')
    # plt.plot(Ypred_mean_gp[:,ix[0]]+Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]+Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')
    # plt.plot(Ypred_mean_gp[:,ix[0]]-Ypred_std_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]]-Ypred_std_gp[:,ix[1]], '--r', label='rollout mean')

    plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
    t = 26
    X = np.concatenate((Ypred_mean_gpup[:t,ix[0]]+Ypred_std_gpup[:t,ix[0]], np.flip(Ypred_mean_gpup[:t,ix[0]]-Ypred_std_gp[:t,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gpup[:t,ix[1]]+Ypred_std_gpup[:t,ix[1]], np.flip(Ypred_mean_gpup[:t,ix[1]]-Ypred_std_gp[:t,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'c')
    t1 = t+17
    X = np.concatenate((Ypred_mean_gpup[t:t1,ix[0]]+Ypred_std_gpup[t:t1,ix[0]], np.flip(Ypred_mean_gpup[t:t1,ix[0]]-Ypred_std_gpup[t:t1,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gpup[t:t1,ix[1]]+Ypred_std_gpup[t:t1,ix[1]], np.flip(Ypred_mean_gpup[t:t1,ix[1]]-Ypred_std_gpup[t:t1,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'c')
    t2 = Ypred_mean_gp.shape[0]
    X = np.concatenate((Ypred_mean_gpup[t1-1:t2,ix[0]]+Ypred_std_gpup[t1-1:t2,ix[0]], np.flip(Ypred_mean_gpup[t1-1:t2,ix[0]]-Ypred_std_gpup[t1-1:t2,ix[0]])), axis=0)
    Y = np.concatenate((Ypred_mean_gpup[t1-1:t2,ix[1]]+Ypred_std_gpup[t1-1:t2,ix[1]], np.flip(Ypred_mean_gpup[t1-1:t2,ix[1]]-Ypred_std_gpup[t1-1:t2,ix[1]])), axis=0)
    plt.fill( X, Y , alpha = 0.5 , color = 'c')
    # plt.plot(Ypred_mean_gpup[:,ix[0]]+Ypred_std_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]]+Ypred_std_gpup[:,ix[1]], '--c', label='rollout mean')
    # plt.plot(Ypred_mean_gpup[:,ix[0]]-Ypred_std_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]]-Ypred_std_gpup[:,ix[1]], '--c', label='rollout mean')

    plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '-k', label='Naive')
    plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')

    ix = [0, 10, 20, 30, 42, 48]
    for i in range(len(ix)):
        plt.plot(Pgp[ix[i]][:,0], Pgp[ix[i]][:,1], '.k')

if tr == '4' or tr == '5':
    plt.plot(Smean[:,ix[0]], Smean[:,ix[1]], '-b', label='rollout mean')
    plt.plot(Smean[:,ix[0]]+Sstd[:,ix[0]], Smean[:,ix[1]]+Sstd[:,ix[1]], '--b', label='rollout mean')
    plt.plot(Smean[:,ix[0]]-Sstd[:,ix[0]], Smean[:,ix[1]]-Sstd[:,ix[1]], '--b', label='rollout mean')

    plt.plot(Ypred_mean_gp[:,ix[0]], Ypred_mean_gp[:,ix[1]], '-r', label='BPP mean')

    plt.plot(Ypred_mean_gpup[:,ix[0]], Ypred_mean_gpup[:,ix[1]], '-c', label='GPUP mean')
    
    plt.plot(Ypred_naive[:,0], Ypred_naive[:,1], '-k', label='Naive')
    plt.plot(Ypred_bmean[:,0], Ypred_bmean[:,1], '-m', label='Batch mean')

plt.show()

