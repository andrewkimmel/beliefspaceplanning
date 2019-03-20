#!/usr/bin/env python

import rospy
import numpy as np
import time
import random
from transition_experience import *
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool
from std_srvs.srv import Empty, EmptyResponse


class actorPubRec():
    discrete_actions = True

    gripper_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    obj_pos = np.array([0., 0.])
    obj_vel = np.array([0., 0.])
    fail = False
    Done = False
    running = False
    action = np.array([0.,0.])
    state = np.array([0.,0., 0., 0.])
    joint_states = np.array([0., 0., 0., 0.])
    joint_velocities = np.array([0., 0., 0., 0.])
    n = 0
    pi_cross = False
    
    texp = transition_experience(Load=True, discrete = discrete_actions, postfix='_bu')

    def __init__(self):
        rospy.init_node('actor_pub_record', anonymous=True)

        rospy.Subscriber('/acrobot/my_joint_states', Float32MultiArray, self.callbackJoints)
        rospy.Subscriber('/acrobot/my_joint_velocities', Float32MultiArray, self.callbackJointsVel)
        rospy.Subscriber('/collect/gripper_action', Float32MultiArray, self.callbackAction)
        rospy.Subscriber('/acrobot/pi_cross', Bool, self.callbackCross)

        rospy.Service('/actor/trigger', Empty, self.callbackTrigger)
        rospy.Service('/actor/save', Empty, self.callbackSave)

        rate = rospy.Rate(20)
        count = 0
        while not rospy.is_shutdown():

            if self.running:
                self.state = np.concatenate((self.joint_states, self.joint_velocities), axis=0)
                
                self.texp.add(self.state, self.action, self.state, self.fail or self.pi_cross)

                if self.fail or self.pi_cross:
                    print('[recorder] Episode ended (%d points so far).' % self.texp.getSize())
                    self.running = False

            rate.sleep()

    def callbackJoints(self, msg):
        self.joint_states = np.array(msg.data)

    def callbackJointsVel(self, msg):
        self.joint_velocities = np.array(msg.data)

        self.fail = True if any(np.abs(self.joint_velocities) >= 12.) else False

    def callbackCross(self, msg):
        self.pi_cross = np.array(msg.data)

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbackTrigger(self, msg):
        self.running = not self.running
        if self.running:
            self.pi_cross = False

        return EmptyResponse()

    def callbackSave(self, msg):
        self.texp.save()

        return EmptyResponse()


if __name__ == '__main__':
    
    try:
        actorPubRec()
    except rospy.ROSInterruptException:
        pass