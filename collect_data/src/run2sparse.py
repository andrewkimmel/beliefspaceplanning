#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import rolloutReq
import time

epi_srv = rospy.ServiceProxy('/collect/planned_episode', rolloutReq)
save_srv = rospy.ServiceProxy('/collect/save_data', Empty)
rand_epi_srv = rospy.ServiceProxy('/collect/random_episode', Empty)


for i in range(1,100000):

    if np.random.uniform() > 1.1:
        # goal = str(np.random.randint(1,18))
        # print('Running goal number ' + goal + '.')

        # A = np.loadtxt('/home/pracsys/catkin_ws/src/beliefspaceplanning/collect_data/example_paths/plan' + goal + '.txt', delimiter=',', dtype=float)

        # j = np.random.randint(10)# if A.shape[0] > 50 else np.random.randint(15)
        # A = A[:-j,:]

        # Af = []
        # for a in A:
        #     for _ in range(10):
        #         Af.append(a)
        # Af = np.array(Af)

        print('Running shooting...')
        a = np.array([-1.,1.]) if np.random.uniform() > 0.5 else np.array([1.,-1.])
        n = np.random.randint(40, 300)
        Af = []
        for _ in range(n):
            Af.append(a)

        Af = np.array(Af)

        #epi_srv(Af.reshape(-1,1))
        if not epi_srv(Af.reshape(-1,1)).success:
            print('Path failed.')
        else:
            print('Path succeeded.')
    else:
        print "Running random episode..."
        rand_epi_srv()
        
    # if not (i % 10):
    #     save_srv()

