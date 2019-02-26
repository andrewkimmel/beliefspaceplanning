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

##########################################################################################################

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
    A = np.array([[-1., 1.] for _ in range(int(200*1./stepSize))])

######################################## Roll-out ##################################################

rospy.init_node('verification_gazebo', anonymous=True)
rate = rospy.Rate(15) # 15hz

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

if 0:
    Af = A.reshape((-1,))
    Pro = []
    for j in range(1):
        print("Rollout number " + str(j) + ".")
        
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


# plt.title('path ' + tr)
# plt.show()
# exit(1)

if 0:
    D = []
    for l in range(20):
        Np = 200 # Number of particles

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
            S = S_next

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

            # s_next = np.ones((1,6))

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

            st = time.time()
            res = gp_srv(S.reshape(-1,1), a)
            t_mean += (time.time() - st) 

            if res.node_probability < p_mean:
                p_mean = res.node_probability
            S_next = np.array(res.next_states).reshape(-1,state_dim)
            s_mean_next = np.mean(S_next, 0)
            S = np.tile(s_mean_next, (Np,1))

            # s_mean_next = np.ones((1,6))

            Ypred_bmean = np.append(Ypred_bmean, s_mean_next.reshape(1,state_dim), axis=0)

        t_mean /= A.shape[0]

        ######################################## GPUP propagation ###############################################
        if l == 0:
            print "Running GPUP."
            sigma_start += np.ones((state_dim,))*1e-4
    
            t_gpup = 0

            s = np.copy(s_start)
            sigma_x = sigma_start
            Ypred_mean_gpup = s.reshape(1,state_dim)
            Ypred_std_gpup = sigma_x.reshape(1,state_dim)

            print("Running (open loop) path...")
            p_gpup = 1.
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

                # s_next = sigma_next = np.ones((1,6))

                Ypred_mean_gpup = np.append(Ypred_mean_gpup, s_next.reshape(1,state_dim), axis=0) #Ypred_mean_gpup,np.array([0,0,0,0]).reshape(1,state_dim),axis=0)#
                Ypred_std_gpup = np.append(Ypred_std_gpup, sigma_next.reshape(1,state_dim), axis=0)

            t_gpup /= A.shape[0]

        ######################################## Compare ###########################################################

        d_gp = d_gpup = d_naive = d_mean = d = 0.
        for i in range(A.shape[0]):
            if i < Smean.shape[0]-1:
                d += np.linalg.norm(Smean[i,:2]-Smean[i+1,:2])
            d_gp += np.linalg.norm(Ypred_mean_gp[i,:2] - Smean[i,:2])
            d_naive += np.linalg.norm(Ypred_bmean[i,:2] - Smean[i,:2])
            d_mean += np.linalg.norm(Ypred_naive[i,:2] - Smean[i,:2])
            d_gpup += np.linalg.norm(Ypred_mean_gpup[i,:2] - Smean[i,:2])
        d_gp = np.sqrt(d_gp/A.shape[0])
        d_naive = np.sqrt(d_naive/A.shape[0])
        d_mean = np.sqrt(d_mean/A.shape[0])
        d_gpup = np.sqrt(d_gpup/A.shape[0])

        ######################################## Save ###########################################################

        D.append((d_gp, d_naive, d_mean, d_gpup, p_gp, p_naive, p_mean, p_gpup, d))

        with open(path + 'stats_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl', 'w') as f:
            pickle.dump(D, f)

######################################## Stats ###########################################################

with open(path + 'stats_' + tr + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(stepSize) + '.pkl') as f:  
    D = pickle.load(f)  

D = np.mean(np.array(D), 0)


print "-----------------------------------"
print "Path length: " + str(D[8]) + " mm"
print "-----------------------------------"
print "Naive rmse: " + str(D[1]) + " mm"
print "mean rmse: "  + str(D[2]) + " mm"
print "GP rmse: "    + str(D[0]) + " mm"
print "GPUP rmse: "  + str(D[3]) + " mm"
print "-----------------------------------"
print "GP naive probability: "   + str(D[5])
print "GP mean probability: "    + str(D[6])
print "GP probability: "         + str(D[4])
print "GPUP probability: "       + str(D[7])
print "-----------------------------------"

