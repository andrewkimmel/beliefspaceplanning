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

for i in range(1,100000):
    goal = str(np.random.randint(1,10))
    print('Running goal number ' + goal + '.')

    A = np.loadtxt('/home/pracsys/catkin_ws/src/beliefspaceplanning/collect_data/example_paths/plan' + goal + '.txt', delimiter=',', dtype=float)

    if goal == '6' or goal == '1':
        j = np.random.randint(80)
        A = A[:-j,:]
    if goal == '2':
        j = np.random.randint(10)
        A = A[:-j,:]

    if not epi_srv(A.reshape(-1,1)).success:
        print('Path ' + goal + ' failed.')
    else:
        print('Path ' + goal + ' succeeded.')
        
    if not (i % 5):
        process_srv()

