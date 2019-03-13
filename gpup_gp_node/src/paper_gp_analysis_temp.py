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

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/'

from data_load import data_load
dl = data_load(K = 50)
Dtest = dl.Qtest[:,:4]

# plt.plot(Dtest[:,0],Dtest[:,1],'.')
# plt.show()

A = {'Up': [-1.,-1], 'Down': [1.,1.], 'Left': [-1.,1.], 'Right': [1.,-1.]}


if 0:
    R = {}
    for key, a in A.iteritems():
        S = []
        i = 0
        for s in Dtest:
            print "Step %i with action "%i + str(a) + " (%s)..."%key
            i += 1

            try:
                res = naive_srv(s.reshape(-1,1), a)
            except:
                continue

            # s_next = np.array(res.next_state)
            std_next = np.array(res.std)
            S.append(std_next)

        R[key] = S
    with open(path + 'std_test.pkl', 'w') as f:
        pickle.dump(R, f)
else:
    with open(path + 'std_test.pkl','r') as f:  
        R = pickle.load(f)  

plt.figure(1, figsize=(15,10))
i = 1
for V, S in R.iteritems():
    ax = plt.subplot(2,2,i)
    i += 1

    S = np.array(S)#[:,:2]
    G = []
    for s in S:
        if np.any(s > 2.):
            continue
        G.append(s)
    G = np.array(G)
    plt.hist(G,25, label = ['x','y','$u_1$','$u_2$'])
    plt.title(V)
    plt.legend()
    plt.xlim([0,1.0])
# plt.savefig(path + 'variance_dist',dpi=300)

# -------------------------------------------------

Anorm = {'Up': [0.,0.], 'Down': [1.,1.], 'Left': [0.,1.], 'Right': [1.,0.]}
X = dl.Xtrain

T = {'Up': [], 'Down': [], 'Left': [], 'Right': []}
for k in T.keys():
    for x in X:
        if np.all(x[4:]==Anorm[k]):
            T[k].append(x[:4])

plt.figure(2, figsize=(15,10))
i = 1
for V, S in T.iteritems():
    ax = plt.subplot(2,2,i)
    i += 1

    S = np.array(S)[:,:2]
    plt.hist2d(S[:,0],S[:,1],100)#, label = ['x','y','$u_1$','$u_2$'])
    plt.title(V)
    plt.colorbar()
    # plt.legend()

# plt.savefig(path + 'action_dist',dpi=300)

plt.show()


