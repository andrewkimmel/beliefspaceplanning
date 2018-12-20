#!/usr/bin/env python

import rospy
import numpy as np
import time

from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from ChaseProbEnv import ChaseEnv, ProbEnv
from SquareEnv import SquareEnv, ProbEnv
from toy_simulator.srv import TargetAngles, IsDropped, observation, transition

SIZE = 1.3

class toy_sim():

    state = np.array([0.0,0.0])
    state_dim = 2
    svm_fail_probability_lower_bound = 0.65


    def __init__(self):
        # self.env = ChaseEnv(size = SIZE, add_noise = True, yfactor=10)
        self.env = SquareEnv(size = SIZE, add_noise = False)
        self.prob = ProbEnv(size = SIZE)

        pub_obj_pos = rospy.Publisher('/toy/obj_pos', Float32MultiArray, queue_size=10)
        msg = Float32MultiArray()

        rospy.Service('/toy/ResetGripper', Empty, self.ResetGripper)
        rospy.Service('/toy/MoveGripper', TargetAngles, self.MoveGripper)
        rospy.Service('/toy/IsObjDropped', IsDropped, self.CheckDropped)
        rospy.Service('/toy/observation', observation, self.GetObservation)
        rospy.Service('/toy/transition', transition, self.Transition)

        print('[toy_node] Ready...')

        rospy.init_node('toy_sim', anonymous=True)

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            msg.data = self.state
            pub_obj_pos.publish(msg)
            # rospy.spin()
            rate.sleep()

    def ResetGripper(self, msg):
        self.state = self.env.reset()
        self.dropped = False

        print('[toy_node] Gripper reset.')

        return EmptyResponse()

    # Predicts the next step by calling the GP class - gets external state (for planner)
    def MoveGripper(self, req):
        sp = self.env.step(req.angles, scale=5)
        print('[toy_node] Current state s: ' + str(self.state) + ", action: " + str(req.angles))
        print('[toy_node] Predicted next state sp: ' + str(sp))
        self.state = sp
        
        # p, _ = self.prob.probability(self.state) 
        # p_draw = np.random.uniform()
        # print(p_draw, p)
        # if p_draw < p: 
        #     print('[toy_node] Gripper fail with probability %f. End of episode.'%p)
        #     self.dropped = True
        #     return {'success': False}

        for i in range(self.state_dim):
            if self.state[i] < -SIZE or self.state[i] > SIZE:
                print('[toy_node] Object out of bounds. End of episode.')
                self.dropped = True
                return {'success': False}

        return {'success': True}

    def GetObservation(self, msg):

        return {'state': self.state}

    # Reports whether dropped the object
    def CheckDropped(self, req):

        return {'dropped': self.dropped}

    def Transition(self, req):

        state = np.array(req.state)
        action = np.array(req.action)

        next_state_mean, next_state_std = self.env.Step(state, action)

        p, _ = self.prob.probability(next_state_mean)

        for i in range(self.state_dim):
            if next_state_mean[i] < -SIZE or next_state_mean[i] > SIZE:
                p = 1.0

        return {'next_state_mean': next_state_mean, 'next_state_std': next_state_std, 'probability': p}

        

if __name__ == '__main__':
    try:
        toy_sim()
    except rospy.ROSInterruptException:
        pass