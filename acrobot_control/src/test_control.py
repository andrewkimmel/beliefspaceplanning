#!/usr/bin/env python

import rospy
from acrobot_control.srv import pathTrackReq
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


rp = .01
r = .02

rospy.wait_for_service('/control')
track_srv = rospy.ServiceProxy('/control', pathTrackReq)

# File = 'naive_with_svm_goal1_run0_traj'
# File = 'robust_particles_pc_goal0_run0_traj'
# File = 'naive_with_svm_goal2_run1_traj'
tr = '3'
File = 'acrobot_ao_rrt_traj' + tr

traj = '/home/pracsys/Dropbox/transfer/transition_data/Acrobot/noiseless_acrobot/no_obstacles/discrete_control/example_paths/acrobot_ao_rrt_traj' + tr + '.txt'  
plan = '/home/pracsys/Dropbox/transfer/transition_data/Acrobot/noiseless_acrobot/no_obstacles/discrete_control/example_paths/acrobot_ao_rrt_plan' + tr + '.txt'  

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/acrobot_control/results/'

# ctr = np.concatenate((C[int(traj[traj.find('goal')+4]), :], np.array([0,0])), axis=0)

S = np.loadtxt(traj, delimiter=',')
n = S.shape[0] - 3
S = S[:n,:]
At = np.loadtxt(plan, delimiter=',')
At[:,1] = (At[:,1]*100).astype(int)

A = []
for a in At:
    for i in range(int(a[1])):
        A.append(a[0])
A = np.array(A)[:n]
# S = medfilter(S, 20)
# S = np.append(S, [ctr], axis=0)

ctr = S[-1,:]

if 0:
    res = track_srv(S.reshape((-1,)),  A.reshape((-1,)), True)
    Scl = np.array(res.real_path).reshape(-1, S.shape[1])
    Acl = np.array(res.actions).reshape(-1, 1)
    success = res.success

    with open(path + 'cl_' + File + '.pkl', 'w') as f: 
        pickle.dump([Scl, Acl, success], f)
else:
    with open(path + 'cl_' + File + '.pkl', 'r') as f: 
        Scl, Acl, success = pickle.load(f)

if 0:
    res = track_srv(S.reshape((-1,)),  A.reshape((-1,)), False)
    Sol = np.array(res.real_path).reshape(-1, S.shape[1])
    Aol = np.array(res.actions).reshape(-1, 1)
    success = res.success

    with open(path + 'ol_' + File + '.pkl', 'w') as f: 
        pickle.dump([Sol, Aol, success], f)
else:
    with open(path + 'ol_' + File + '.pkl', 'r') as f: 
        Sol, Aol, success = pickle.load(f)

## Plot ##

# print Scl.shape
# print Acl.shape

# S = Scl[800:1200,:]
# A = Acl[800:1200,:]

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

# Obs = np.array([[33, 110, 4.], [-27, 118, 2.5]])

plt.figure(1)
ax1 = plt.subplot(1,2,1)
goal = plt.Circle((ctr[0], ctr[1]), r, color='m')
ax1.add_artist(goal)
goal_plan = plt.Circle((ctr[0], ctr[1]), rp, color='w')
ax1.add_artist(goal_plan)

# for o in Obs:
#     obs = plt.Circle(o[:2], o[2])#, zorder=10)
#     ax.add_artist(obs)

plt.plot(S[:,0],S[:,1],'.-r', label='reference')
plt.plot(S[-1,0],S[-1,1],'dm', label='reference')
plt.plot(Sol[:,0],Sol[:,1],'.-b', label='Rolled-out')
plt.plot(Scl[:,0],Scl[:,1],'.-k', label='Closed-loop')
plt.title(File + ", success: " + str(success))
plt.legend()
plt.axis('equal')
plt.savefig(path + File, dpi=300)

ax = plt.subplot(1,2,2)
plt.plot(S[:,2],S[:,3],'.-r', label='reference')
plt.plot(S[-1,2],S[-1,3],'dm', label='reference')
plt.plot(Sol[:,2],Sol[:,3],'.-b', label='Rolled-out')
plt.plot(Scl[:,2],Scl[:,3],'.-k', label='Closed-loop')
plt.title(File + ", success: " + str(success))
plt.legend()
plt.axis('equal')
# plt.savefig(path + File, dpi=300)

# plt.figure(2)
# plt.plot(S[:-1,2],S[:-1,3],'-r', label='reference')
# plt.plot(Scl[:,2],Scl[:,3],'-k', label='Rolled-out')

plt.figure(3)
ax = plt.subplot(2,2,1)
plt.plot(S[:,0], '--r', label='reference')
plt.plot(Sol[:,0], 'b', label='Rolled-out')
plt.plot(Scl[:,0], 'k', label='closed-loop')
plt.ylabel('x')
plt.xlabel('time')

ax = plt.subplot(2,2,2)
plt.plot(S[:,1], '--r', label='reference')
plt.plot(Sol[:,1], 'b', label='Rolled-out')
plt.plot(Scl[:,1], 'k', label='closed-loop')
plt.ylabel('x')
plt.xlabel('time')

ax = plt.subplot(2,2,3)
plt.plot(A, '--r', label='reference')
plt.plot(Acl, 'b', label='closed-loop')
plt.ylabel('Actions')
plt.xlabel('time')






plt.show()

