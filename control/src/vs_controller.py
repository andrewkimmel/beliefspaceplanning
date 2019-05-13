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
    obj_pos_prev = np.array([0., 0.])
    action = np.array([0.,0.])
    goal = np.array([0.,0.])
    goal_prev = np.array([0.,0.])
    A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])
    vel_magnitude = 1.6
    J = np.array([[1., 1.], [-1., 1.]])

    def __init__(self):
        rospy.init_node('vs_controller', anonymous=True)

        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)

        pub_best_action = rospy.Publisher('/vs_controller/action', Float32MultiArray, queue_size=10)
        pub_2record = rospy.Publisher('/rollout/gripper_action', Float32MultiArray, queue_size=10)
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
        goal = self.goal[:2]

        v = goal - self.obj_pos
        v = v / np.linalg.norm(v)
        v *= self.vel_magnitude
        v[1] *= -1

        # Kp = 2.8
        # Kd = 6.5
        # v *= Kp * np.linalg.norm(goal - self.obj_pos) - Kd * np.linalg.norm(self.obj_pos - self.obj_pos_prev)

        action = np.dot( self.J, v) 

        # a = goal; b = self.goal_prev; s = self.obj_pos
        # d = np.abs( (b[1]-a[1])*s[0]-(b[0]-a[0])*s[1] + b[0]*a[1]-b[1]*a[0] ) / np.sqrt( (b[1]-a[1])**2 + (b[0]-a[0])**2 )
        # print d #a, b, s, d
        # gain = (1.0 if np.isnan(d) or d < 1.7 or np.all(b == 0) else Kp * d) - Kd * np.linalg.norm(self.obj_pos - self.obj_pos_prev)

        # # print "Distance from path: " + str(gain), d, goal, self.goal_prev
        # # e = np.linalg.norm(goal - self.obj_pos)
        # # gain = 0.45 if e < 2.5 else Kp * e# np.exp(0.35 * e)
        # action *= gain

        # self.obj_pos_prev = np.copy(self.obj_pos)

        # action = np.round(action)
        # action[action > 1.] = 1.0
        # action[action < -1.] = -1.0

        # action[action > 3.] = 3.0
        # action[action < -3.] = -3.0

        return action

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def callbackGoal(self, msg):
        if np.any(self.goal != np.array(msg.data)):
            self.goal_prev = np.copy(self.goal[:2])
            self.goal = np.array(msg.data)

if __name__ == '__main__':
    try:
        vs_controller()
    except rospy.ROSInterruptException:
        pass