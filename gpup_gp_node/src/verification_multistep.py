#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import gpup_transition, batch_transition
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import pickle
from rollout_node.srv import rolloutReq
from gp_sim_node.srv import sa_bool

np.random.seed(10)

state_dim = 4+2

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transition', gpup_transition)
svm_srv = rospy.ServiceProxy('/svm_fail_check', sa_bool)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
plot_srv = rospy.ServiceProxy('/rollout/plot', Empty)


tr = '2'
stepSize = 10

##########################################################################################################
if tr == '1':
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

######################################## Roll-out ##################################################

rospy.init_node('verification_multistep', anonymous=True)
rate = rospy.Rate(15) # 15hz

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/verf/'


if 1:
    Af = A.reshape((-1,))
    Pro = []
    for j in range(1):
        print("Rollout number " + str(j) + ".")
        
        Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)

        Pro.append(Sro)

        with open(path + 'ver_rollout_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(Pro, f)

with open(path + 'ver_rollout_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl') as f:  
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
sigma_start = np.std(np.array(S), 0) + np.array([0.,0.,1e-4,1e-4,1e-4,1e-4])
# ax.plot(s_start_mean[0], s_start_mean[1], 'om')
# patch = Ellipse(xy=(s_start[0], s_start[1]), width=sigma_start[0]*2, height=sigma_start[1]*2, angle=0., animated=False, edgecolor='r', linewidth=2., linestyle='-', fill=True)
# ax.add_artist(patch)

Smean = Pro[0]

print("Roll-out success rate: " + str(float(c) / len(Pro)*100) + "%")

plt.show()
exit(1)

######################################## GP propagation ##################################################
Np = 500 # Number of particles
if 1:
    print "Running GP."

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

        print res.node_probability

        Ypred_mean_gp = np.append(Ypred_mean_gp, s_mean_next.reshape(1,state_dim), axis=0)
        Ypred_std_gp = np.append(Ypred_std_gp, s_std_next.reshape(1,state_dim), axis=0)

    ######################################## Save ###########################################################

    with open(path + 'ver_pred_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl', 'w') as f:
        pickle.dump([Ypred_mean_gp, Ypred_std_gp, Pgp], f)

######################################## Plot ###########################################################

with open(path + 'ver_pred_' + tr + '_v5_d6_m' + str(stepSize) + '.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Pgp = pickle.load(f)  

with open(path + 'ver_gf_' + tr + '_v2.pkl') as f:  
    Ig, If = pickle.load(f)  

prtc_mean_line, = ax.plot([], [], '-g')
prtc, = ax.plot([], [], '.k', markersize=1)

sm, = ax.plot([], [], 'ok', markerfacecolor='r', markersize=8)

prtc_mean, = ax.plot([], [], '*g')
patch_prtc = Ellipse(xy=(Ypred_mean_gp[0,0], Ypred_mean_gp[0,1]), width=Ypred_std_gp[0,0]*2, height=Ypred_std_gp[0,1]*2, angle=0., animated=True, edgecolor='y', linewidth=2., fill=False)
ax.add_patch(patch_prtc)

# plt.xlim(np.min(Ypred_mean_gp, 0)[0]*0-5, np.max(Ypred_mean_gp, 0)[0]*1.0)
# plt.ylim(np.min(Ypred_mean_gp, 0)[1]*0.99, np.max(Ypred_mean_gp, 0)[1]*1.01)

# plt.xlim(np.min(Ypred_mean_gp, 0)[0]*1.1, np.max(Ypred_mean_gp, 0)[0]*1.0)
# plt.ylim(np.min(Ypred_mean_gp, 0)[1]*1., np.max(Ypred_mean_gp, 0)[1]*1.04)

def init():
    prtc.set_data([], [])
    prtc_mean.set_data([], [])
    prtc_mean_line.set_data([], [])
    sm.set_data([], [])

    return sm, prtc, prtc_mean, prtc_mean_line, patch_prtc,

def animate(i):

    S = Pgp[i]
    prtc.set_data(S[:,0], S[:,1])

    sm.set_data(Smean[i][0], Smean[i][1])

    prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    prtc_mean_line.set_data(Ypred_mean_gp[:i+1,0], Ypred_mean_gp[:i+1,1])

    patch_prtc.center = (Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    patch_prtc.width = Ypred_std_gp[i,0]*2
    patch_prtc.height = Ypred_std_gp[i,1]*2

    return sm, prtc, prtc_mean, prtc_mean_line, patch_prtc,

ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=50, repeat_delay=1000, blit=True)
# ani.save(path + 'belief_gazebo_' + str(tr) + '_v5_d6_m' + str(stepSize) + '.mp4', metadata={'artist':'Avishai Sintov','year':'2019'}, bitrate=-1, codec="libx264")

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

