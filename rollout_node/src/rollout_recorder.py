#!/usr/bin/env python

import rospy
import numpy as np
import time
import random
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool
from std_srvs.srv import Empty, EmptyResponse, SetBool
from rollout_node.srv import gets

class rolloutRec():
    discrete_actions = True

    Done = False
    running = False
    action = np.array([0.,0.])
    state = np.array([0.,0., 0., 0.])
    joint_states = np.array([0., 0.])
    joint_velocities = np.array([0., 0.])
    pi_cross = False
    fail = False
    n = 0
    S = []
    A = []
    
    def __init__(self):
                
        rospy.init_node('rollout_recorder', anonymous=True)

        rospy.Subscriber('/rollout/gripper_action', Float32MultiArray, self.callbackAction)
        rospy.Subscriber('/acrobot/my_joint_states', Float32MultiArray, self.callbackJoints)
        rospy.Subscriber('/acrobot/my_joint_velocities', Float32MultiArray, self.callbackJointsVel)
        rospy.Subscriber('/acrobot/pi_cross', Bool, self.callbackCross)

        rospy.Service('/rollout_recorder/trigger', SetBool, self.callbackTrigger)
        rospy.Service('/rollout_recorder/get_states', gets, self.get_states)

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():

            if self.running:
                self.state = np.concatenate((self.joint_states, self.joint_velocities), axis=0)

                self.S.append(self.state)
                self.A.append(self.action)
                
                if self.pi_cross or self.fail:
                    print('[rollout_recorder] Episode ended.')
                    self.running = False

            rate.sleep()

    def callbackJoints(self, msg):
        self.joint_states = np.array(msg.data)

    def callbackJointsVel(self, msg):
        self.joint_velocities = np.array(msg.data)

        self.fail = True if any(np.abs(self.joint_velocities) >= 12.) else False

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbackCross(self, msg):
        self.pi_cross = np.array(msg.data)

    def callbackTrigger(self, msg):
        self.running = msg.data
        if self.running:
            self.S = []
            self.A = []

        return {'success': True, 'message': ''}

    def get_states(self, msg):

        return {'states': np.array(self.S).reshape((-1,)), 'actions': np.array(self.A).reshape((-1,))}

       
if __name__ == '__main__':
    
    try:
        rolloutRec()
    except rospy.ROSInterruptException:
        pass