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
tr = '6'
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
    D = []
    for l in range(10):
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
        if l == 0:
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

        ######################################## Compare ###########################################################

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
        print d

        ######################################## Save ###########################################################

        D.append((d_gp, d_naive, d_mean, d_gpup, p_gp, p_naive, p_mean, p_gpup))

        with open(path + 'stats_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl', 'w') as f:
            pickle.dump(D, f)

######################################## Stats ###########################################################


with open(path + 'stats_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl') as f:  
    D = pickle.load(f)  

# print np.array(D)[:,7]

D = np.mean(np.array(D), 0)


# print "-----------------------------------"
# print "Path length: " + str(d)
print "-----------------------------------"
print "Naive rmse: " + str(D[1]) + "mm"
print "mean rmse: " + str(D[2]) + "mm"
print "GP rmse: " + str(D[0]) + "mm"
print "GPUP rmse: " + str(D[3]) + "mm"
print "-----------------------------------"
print "GP naive probability: " + str(D[5])
print "GP mean probability: " + str(D[6])
print "GP probability: " + str(D[4])
print "GPUP probability: " + str(D[7])
print "-----------------------------------"

