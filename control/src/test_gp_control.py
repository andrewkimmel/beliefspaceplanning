#!/usr/bin/env python

import rospy
from control.srv import pathTrackReq
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import pickle
import time

# np.random.seed(10)

def medfilter(X, W):
    w = int(W/2)

    for k in range(X.shape[1]):
        x = np.copy(X[:,k])
        x_new = np.copy(x)
        for i in range(1, x.shape[0]-1):
            if i < w:
                x_new[i] = np.mean(x[:i+w])
            elif i > x.shape[0]-w:
                x_new[i] = np.mean(x[i-w:])
            else:
                x_new[i] = np.mean(x[i-w:i+w])
        X[:,k] = np.copy(x_new)
    return X


rp = 7.
r = 10.
C = np.array([
        [-37, 119],
        [-33, 102],
        [-22, 129],
        [-52, 112],
        [67, 80],
        [-63, 91]])

track_srv = rospy.ServiceProxy('/control', pathTrackReq)

# File = 'naive_with_svm_goal1_run0_traj'
# File = 'robust_particles_pc_goal0_run0_traj'
# File = 'naive_with_svm_goal2_run1_traj'
File = 'robust_particles_pc_goal4_run0_traj'

traj = '/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/set4/' + File + '.txt' 
path = '/home/juntao/catkin_ws/src/beliefspaceplanning/control/results/'

ctr = np.concatenate((C[int(traj[traj.find('goal')+4]), :], np.array([0,0])), axis=0)

S = np.loadtxt(traj, delimiter=',')
S = medfilter(S, 20)
S = np.append(S, [ctr], axis=0)

if 1:
    res = track_srv(S.reshape((-1,)))
    Sreal = np.array(res.real_path).reshape(-1, S.shape[1])
    Areal = np.array(res.actions).reshape(-1, 2)
    success = res.success

    with open(path + 'control_' + File + '.pkl', 'w') as f: 
        pickle.dump([Sreal, Areal, S, success], f)
else:
    with open(path + 'control_' + File + '.pkl', 'r') as f: 
        Sreal, Areal, S, success = pickle.load(f)

## Plot ##

# print Sreal.shape
# print Areal.shape

# S = Sreal[800:1200,:]
# A = Areal[800:1200,:]

# from gpup_gp_node.srv import one_transition
# gp = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

# i = 0
# for s, a in zip(S, A):
#     print i, s, a
#     res = gp(s.reshape(-1,1), a)
#     # s_next = np.array(res.next_state)
#     # cur_s = np.copy(s_next)
#     i += 1

# exit(1)

Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])

plt.figure(1)
ax = plt.subplot()
goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
ax.add_artist(goal)
goal_plan = plt.Circle((ctr[0], ctr[1]), rp, color='w')
ax.add_artist(goal_plan)

for o in Obs:
    obs = plt.Circle(o[:2], o[2])#, zorder=10)
    ax.add_artist(obs)

plt.plot(S[:-1,0],S[:-1,1],'-r', label='reference')
plt.plot(Sreal[:,0],Sreal[:,1],'-k', label='Rolled-out')
plt.title(File + ", success: " + str(success))
plt.legend()
plt.axis('equal')
plt.savefig(path + File, dpi=300)

plt.figure(2)
plt.plot(S[:-1,2],S[:-1,3],'-r', label='reference')
plt.plot(Sreal[:,2],Sreal[:,3],'-k', label='Rolled-out')

plt.show()


