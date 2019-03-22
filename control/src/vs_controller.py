#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import pickle
from control.srv import pathTrackReq

import sys
sys.path.insert(0, '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class vs_controller():

    drop = True
    obj_pos = np.array([0., 0.])
    action = np.array([0.,0.])
    goal = np.array([0.,0.,0.,0.])
    A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])
    vel_magnitude = 1.0
    J = np.array([[1., 1.], [-1., 1.]])

    def __init__(self):
        rospy.init_node('vs_controller', anonymous=True)

        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)

        pub_best_action = rospy.Publisher('/vs_controller/action', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/control/goal', Float32MultiArray, self.callbackGoal)

        msg = Float32MultiArray()

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.action = self.get_action()

            msg.data = self.action
            pub_best_action.publish(msg)
            rate.sleep()

    def get_action(self):
        goal = self.goal[:2]

        v = goal - self.obj_pos
        v = v / np.linalg.norm(v)
        v *= self.vel_magnitude
        v[1] *= -1

        action = np.dot( self.J, v) 

        return action

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)


if __name__ == '__main__':
    try:
        vs_controller()
    except rospy.ROSInterruptException:
        pass