#!/usr/bin/env python

import rospy
import numpy as np
import time

from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16, Bool
from std_srvs.srv import SetBool, Empty, EmptyResponse
from SquareEnv import SquareEnv
from toy_simulator.srv import TargetAngles, IsDropped, observation, transition

class toy_sim():

    state = np.array([0.0,0.0])
    state_dim = 2

    dropped = False

    def __init__(self):
        self.env = SquareEnv()

        pub_obj_pos = rospy.Publisher('/toy/obj_pos', Float32MultiArray, queue_size=10)
        pub_drop = rospy.Publisher('/toy/cylinder_drop', Bool, queue_size=10)
        msg = Float32MultiArray()

        rospy.Service('/toy/ResetGripper', Empty, self.ResetGripper)
        rospy.Service('/toy/MoveGripper', TargetAngles, self.MoveGripper)
        rospy.Service('/toy/IsObjDropped', IsDropped, self.CheckDropped)
        rospy.Service('/toy/observation', observation, self.GetObservation)

        rospy.Service('/toy/plot', Empty, self.callbackPlot)

        print('[toy_node] Ready...')

        rospy.init_node('toy_sim', anonymous=True)

        self.rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            msg.data = self.state
            pub_obj_pos.publish(msg)

            pub_drop.publish(self.dropped)
            # rospy.spin()
            self.rate.sleep()

    def ResetGripper(self, msg):
        self.state = self.env.reset()
        self.dropped = False
        self.rate.sleep()

        print('[toy_node] Gripper reset.')

        return EmptyResponse()

    # Predicts the next step by calling the GP class - gets external state (for planner)
    def MoveGripper(self, req):
        self.state, self.dropped  = self.env.step(req.angles)

        return {'success': not self.dropped}

    def GetObservation(self, msg):

        return {'state': self.state}

    # Reports whether dropped the object
    def CheckDropped(self, req):

        return {'dropped': self.dropped}

    def callbackPlot(self, req):

        self.env.plot()


if __name__ == '__main__':
    try:
        toy_sim()
    except rospy.ROSInterruptException:
        pass