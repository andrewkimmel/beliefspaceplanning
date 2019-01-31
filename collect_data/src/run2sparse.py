#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import rolloutReq
import time

epi_srv = rospy.ServiceProxy('/collect/planned_episode', rolloutReq)
process_srv = rospy.ServiceProxy('/collect/process_data', Empty)
rand_epi_srv = rospy.ServiceProxy('/collect/random_episode', Empty)


# obj 1514016
# mat 1182011

for i in range(1,100000):

    if np.random.uniform() > 0.5:
        goal = str(np.random.randint(1,18))
        print('Running goal number ' + goal + '.')

        A = np.loadtxt('/home/pracsys/catkin_ws/src/beliefspaceplanning/collect_data/example_paths/plan' + goal + '.txt', delimiter=',', dtype=float)

        j = np.random.randint(50) if A.shape[0] > 50 else np.random.randint(15)
        A = A[:-j,:]

        if not epi_srv(A.reshape(-1,1)).success:
            print('Path ' + goal + ' failed.')
        else:
            print('Path ' + goal + ' succeeded.')
    else:
        print "Running random episode."
        rand_epi_srv()

        
    if not (i % 5):
        process_srv()

