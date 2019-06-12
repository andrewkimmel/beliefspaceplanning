#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import rolloutReq
import time
import random

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

state_dim = 8
stepSize = 4#var.stepSize_
version = var.data_version_

action_seq = []
# 0
A = np.tile(np.array([-1.,1.]), (800*1/stepSize,1))
action_seq.append(A)
# 1
A = np.concatenate(  (np.array([[1.,  -1.] for _ in range(int(700*1./stepSize))]), 
        np.array([[ -1., 1.] for _ in range(int(200*1./stepSize))]),
        np.array([[ 1., 1.] for _ in range(int(150*1./stepSize))]),
        np.array([[ 0, -1.] for _ in range(int(200*1./stepSize))]),
        ), axis=0 )
action_seq.append(A)
# 2
A = np.concatenate( (np.array([[-1.,  1.] for _ in range(int(400*1./stepSize))]), 
        np.array([[ 1., -1.] for _ in range(int(400*1./stepSize))]),
        np.array([[-1., 1.] for _ in range(int(400*1./stepSize))]) ), axis=0 )
action_seq.append(A)
# 3
A = np.concatenate( (np.array([[1.,  -1.] for _ in range(int(200*1./stepSize))]), 
        np.array([[1.,  1.] for _ in range(int(200*1./stepSize))]), 
        np.array([[ -1., 1.] for _ in range(int(500*1./stepSize))])), axis=0 )
action_seq.append(A)
# 4
A = np.concatenate( (np.array([[-1.,  1.] for _ in range(int(200*1./stepSize))]), 
        np.array([[1.,  1.] for _ in range(int(170*1./stepSize))]), 
        np.array([[ 1., -1.] for _ in range(int(250*1./stepSize))]),
        np.array([[ -1., -1.] for _ in range(int(250*1./stepSize))])), axis=0 )
action_seq.append(A)
# 5
A = np.concatenate(  (np.array([[-1.,  1.] for _ in range(int(600*1./stepSize))]), 
        np.array([[1.5,  0.] for _ in range(int(300*1./stepSize))]), 
        np.array([[-1.,  -1.] for _ in range(int(120*1./stepSize))]), 
        np.array([[0.0,  1.] for _ in range(int(300*1./stepSize))]), 
        np.array([[ 0., -1.] for _ in range(int(200*1./stepSize))])), axis=0 )
action_seq.append(A)

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
rospy.init_node('collect_test_paths', anonymous=True)

Obj = 'cyl19'
path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'

if 1:
    test_paths = []
    Suc = []
    i = 0
    for A in action_seq:
        Af = A.reshape((-1,))
        print("Rollout number " + str(i) + ".")
        i += 1
        
        roll = rollout_srv(Af, np.array([0.,0.,0.,0.]))
        S = np.array(roll.states).reshape(-1,state_dim)
        suc = roll.success
        print("Got %d points with a %s trial."%(S.shape[0], 'successful' if suc else 'failed'))

        test_paths.append(S)
        Suc.append(suc)

    with open(path + 'testpaths_' + Obj + '_d_v' + str(version) + '.pkl', 'w') as f: 
        pickle.dump([action_seq, test_paths, Obj, Suc], f)
else:
    with open(path + 'testpaths_' + Obj + '_d_v' + str(version) + '.pkl', 'r') as f: 
        action_seq, test_paths, Obj, Suc = pickle.load(f)


plt.figure(1)
plt.title('Object position')
i = 0
for S in test_paths:
    plt.plot(S[:,0], S[:,1], color=(random.random(), random.random(), random.random()), label='path ' + str(i))
    i += 1
plt.legend()

plt.figure(2)
plt.title('Angle')
i = 0
for S in test_paths:
    plt.plot(np.rad2deg(S[:,2]), color=(random.random(), random.random(), random.random()), label='path ' + str(i))
    i += 1
plt.legend()
plt.show()




