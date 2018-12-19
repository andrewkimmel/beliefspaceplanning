#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from gp_sim_node.srv import transition
import math
import numpy as np

import matlab.engine

global eng
global gp_obj

predict_mode = 1

class Spin_gp_hand_sim():

    # A = np.array([[1., 1], [-1., 1.], [1., -1.], [1., 1.], [0, 1.], [1., 0], [0, -1.], [-1., 0]])
    # state_region = np.array([[-87.7413   -5.8484    2.1900    0.4800], [93.3154  143.0419   66.0309   79.8304]]) # Gazebo simulation

    def __init__(self):

        rospy.Service('/transition', transition, self.GetTransition)

        msg = Float32MultiArray()

        rospy.init_node('gp_transition_sim', anonymous=True)

        rate = rospy.Rate(15) # 15hz
        while not rospy.is_shutdown():
            rospy.spin()
            # rate.sleep()      

    # Predicts the next step by calling the GP class - gets external state (for planner)
    def GetTransition(self, req):
        
        # matS = []
        matS = matlab.double(req.state)
        matA = matlab.double(req.action)

        print('[gp_hand_sim] Current state s: ' + str(matS) + ", action: " + str(matA))

        sp, sigma = eng.predict(gp_obj, matS, matA, nargout=2)

        print('[gp_hand_sim] Predicted next state sp: ' + str(sp[0]) + ", sigma: " + str(sigma[0]))

        return {'next_state_mean': sp[0], 'next_state_std': sigma[0]}

        
if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    eng.addpath('/home/pracsys/catkin_ws/src/beliefspaceplanning/gp_sim_node/src')
    gp_obj = eng.gp_class(predict_mode, True)
    try:
        SP = Spin_gp_hand_sim()
    except rospy.ROSInterruptException:
        pass