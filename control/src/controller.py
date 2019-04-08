#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import pickle
from control.srv import pathTrackReq

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class vs_controller():

    drop = True
    state = np.array([0., 0., 0., 0.])
    action = np.array([0.])
    goal = np.array([0., 0., 0., 0.])

    def __init__(self):
        rospy.init_node('vs_controller', anonymous=True)

        rospy.Subscriber('/acrobot/state', Float32MultiArray, self.callbackState)
        pub_best_action = rospy.Publisher('/controller/action', Float32MultiArray, queue_size=10)
        pub_2record = rospy.Publisher('/acrobot/action', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/control/goal', Float32MultiArray, self.callbackGoal)

        msg = Float32MultiArray()

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.action = self.get_action()

            msg.data = self.action
            pub_best_action.publish(msg)
            pub_2record.publish(msg)
            rate.sleep()

    def get_action(self):

        e = self.goal - self.state
        K = np.array([1., 1., 1., 1.])

        action = np.dot( K, e) 

        return action

    def callbackState(self, msg):
        self.state = np.array(msg.data)

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)

if __name__ == '__main__':
    try:
        vs_controller()
    except rospy.ROSInterruptException:
        pass