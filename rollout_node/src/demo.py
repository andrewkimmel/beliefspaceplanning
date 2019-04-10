#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import rolloutReq
import time
import glob
from scipy.io import loadmat


rollout = 0

# path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'
# path = '/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'

# comp = 'juntao'
comp = 'pracsys'

rospy.init_node('demo', anonymous=True)


rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
move_lift_srv = rospy.ServiceProxy('/LiftHand', Empty)
reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)

reset_srv()

A = np.array([[2., -2.] for _ in range(int(100*1./10))])
rollout_srv(A.reshape((-1,)))

move_lift_srv()
rospy.sleep(3.)
move_lift_srv()

A = np.concatenate( (np.array([[ -2., 2.] for _ in range(int(130*1./10))]), 
            np.array([[ 2., 2.] for _ in range(int(50*1./10))]) ), axis=0 )
rollout_srv(A.reshape((-1,)))
move_lift_srv()






        
        

    