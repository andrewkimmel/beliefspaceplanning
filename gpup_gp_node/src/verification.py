#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import gpup_transition, batch_transition, gpup_transition_repeat, batch_transition_repeat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import pickle


np.random.seed(10)

state_dim = 4

gp_srv = rospy.ServiceProxy('/gp/transition', batch_transition)
gpup_srv = rospy.ServiceProxy('/gpup/transitionRepeat', gpup_transition_repeat)
msg_gp = batch_transition_repeat()
msg_gpup = gpup_transition_repeat()

##########################################################################################################

actions = np.concatenate( (np.array([[-0.2, 0.2] for _ in range(150)]), np.array([[-0.2, -0.2] for _ in range(150)]), np.array([[0.2, -0.2] for _ in range(150)]), np.array([[0.2, 0.2] for _ in range(150)]) ), axis=0 )

s_start = np.array([33.4020000000000,-325.930000000000,52,-198])
sigma_start = np.array([0., 0., 0., 0.])+1e-8

######################################## GP propagation ##################################################
if 1:
    print "Running GP."

    Np = 100 # Number of particles
    s = s_start
    S = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np,state_dim))
    Ypred_mean_gp = s.reshape(1,state_dim)
    Ypred_std_gp = np.zeros((1,state_dim)).reshape(1,state_dim)

    Pgp = []; 
    print("Running (open loop) path...")
    for i in range(0, 5+0*actions.shape[0]):
        print("Step " + str(i) + " of " + str(actions.shape[0]))
        Pgp.append(S)
        a = actions[i,:]

        res = gp_srv(S.reshape(-1,1), a, 10)
        S_next = np.array(res.next_states).reshape(-1,state_dim)
        s_mean_next = np.mean(S_next, 0)
        s_std_next = np.std(S_next, 0)
        S = S_next

        Ypred_mean_gp = np.append(Ypred_mean_gp, s_mean_next.reshape(1,state_dim), axis=0)
        Ypred_std_gp = np.append(Ypred_std_gp, s_std_next.reshape(1,state_dim), axis=0)

    ######################################## GPUP propagation ###############################################

    print "Running GPUP."

    s = s_start
    sigma_x = sigma_start
    Ypred_mean_gpup = s.reshape(1,state_dim)
    Ypred_std_gpup = sigma_x.reshape(1,state_dim)

    print("Running (open loop) path...")
    for i in range(0, 5+0*actions.shape[0]):
        print("Step " + str(i) + " of " + str(actions.shape[0]))
        a = actions[i,:]

        res = gpup_srv(s, sigma_x, a, 10)
        s_next = np.array(res.next_mean)
        sigma_next = np.array(res.next_std)
        s = s_next
        sigma_x = sigma_next

        Ypred_mean_gpup = np.append(Ypred_mean_gpup, s_next.reshape(1,state_dim), axis=0)
        Ypred_std_gpup = np.append(Ypred_std_gpup, sigma_next.reshape(1,state_dim), axis=0)

    ######################################## Save ###########################################################

    # with open('belief_ros.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp], f)

######################################## Plot ###########################################################

with open('belief_ros.pkl') as f:  
    Ypred_mean_gp, Ypred_std_gp, Ypred_mean_gpup, Ypred_std_gpup, Pgp = pickle.load(f)  

fig = plt.figure(0)
ax = fig.add_subplot(111)#, aspect='equal')
# ax.plot(Xtest[:,0], Xtest[:,1], 'b-')

prtc_mean_line, = ax.plot([], [], '-r')
prtc, = ax.plot([], [], '.k')
prtc_mean, = ax.plot([], [], '*r')
patch_prtc = Ellipse(xy=(Ypred_mean_gp[0,0], Ypred_mean_gp[0,1]), width=Ypred_std_gp[0,0]*2, height=Ypred_std_gp[0,1]*2, angle=0., animated=True, edgecolor='y', linewidth=2., fill=False)
ax.add_patch(patch_prtc)

patch = Ellipse(xy=(Ypred_mean_gpup[0,0], Ypred_mean_gpup[0,1]), width=Ypred_std_gpup[0,0]*2, height=Ypred_std_gpup[0,1]*2, angle=0., animated=True, edgecolor='m', linewidth=2., linestyle='--', fill=False)
ax.add_patch(patch)
patch_mean, = ax.plot([], [], '--m')

plt.xlim(np.min(Ypred_mean_gp, 0)[0]*0.8, np.max(Ypred_mean_gp, 0)[0]*1.2)
plt.ylim(np.min(Ypred_mean_gp, 0)[1]*0.95, np.max(Ypred_mean_gp, 0)[1]*1.05)

def init():
    prtc.set_data([], [])
    prtc_mean.set_data([], [])
    prtc_mean_line.set_data([], [])
    patch_mean.set_data([], [])

    return prtc, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

def animate(i):

    S = Pgp[i]
    prtc.set_data(S[:,0], S[:,1])
    prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    prtc_mean_line.set_data(Ypred_mean_gp[:i,0], Ypred_mean_gp[:i,1])
    patch_mean.set_data(Ypred_mean_gpup[:i,0], Ypred_mean_gpup[:i,1])

    patch_prtc.center = (Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    patch_prtc.width = Ypred_std_gp[i,0]*2
    patch_prtc.height = Ypred_std_gp[i,1]*2

    patch.center = (Ypred_mean_gpup[i,0], Ypred_mean_gpup[i,1])
    patch.width = Ypred_std_gpup[i,0]*2
    patch.height = Ypred_std_gpup[i,1]*2

    return prtc, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=10, repeat_delay=100, blit=True)
# ani.save('belief2.mp4', metadata={'artist':'Avishai Sintov','year':'2019'})

plt.figure(1)
plt.plot(Ypred_std_gp[:,0], Ypred_std_gp[:,1],'.-y')
plt.plot(Ypred_std_gpup[:,0], Ypred_std_gpup[:,1],'.-m')

# plt.plot(Ypred_mean_gp[:,0], Ypred_mean_gp[:,1],'.-y')
# plt.plot(Ypred_mean_gpup[:,0], Ypred_mean_gpup[:,1],'.-m')

plt.show()

