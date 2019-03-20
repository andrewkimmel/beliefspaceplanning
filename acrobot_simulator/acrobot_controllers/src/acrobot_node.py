#!/usr/bin/python 

'''
----------------------------
Author: Avishai Sintov
        Rutgers University
Date: March 2019
----------------------------
'''


import rospy
import numpy as np 
from std_msgs.msg import Float64, String
from acrobot_controllers.srv import moveservo
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Float32
from std_srvs.srv import Empty, EmptyResponse

class SimAcrobotNode():

    act_load = 0.
    act_torque = 0.
    act_angles = np.array([0,0]) # actuator angles of all three fingers
    # fingers_angles = np.array([0.,0.,0.,0.]) # left_proximal, left_distal, right_proximal, right_distal
    act_vel = np.array([0,0]) # left_proximal, left_distal, right_proximal, right_distal
    # ref_angles = np.array([0.,0.,0.,0.,0.,0.]) # spring reference angle left_proximal, left_distal, right_proximal, right_distal   

    dc = 0.025
    max_load = 0.5
    
    def __init__(self):
        rospy.init_node('SimAcrobotNode', anonymous=True)

        #initialize service handlers:
        rospy.Service('MoveServos', moveservo, self.MoveServosProxy)

        rospy.Subscriber('/acrobot/my_joint_states', Float32MultiArray, self.JointStatesCallback)
        rospy.Subscriber('/acrobot/my_joint_velocities', Float32MultiArray, self.JointVelCallback)

        self.pub_j1 = rospy.Publisher('/acrobot/joint1_controller/command', Float64, queue_size=10)

        self.gripper_load_pub = rospy.Publisher('/acrobot/load', Float32, queue_size=10)
        msg = Float32()

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():

            d = self.set_damping_matrix()
            self.act_torque = self.act_load*self.max_load - d * self.act_vel[0] # Damping does not work well because of numerical errors

            self.pub_j1.publish(self.act_torque)
            
            msg.data = self.act_load
            self.gripper_load_pub.publish(msg)
            
            rate.sleep()

    def set_damping_matrix(self):
        # if np.abs(self.act_vel[0]) < 0.2e-2:
        #     d = 0.0
        # else:
        #     if self.act_angles[0] > 0 and self.act_angles[0] < 2.5:
        #         d = 1.23*self.dc
        #     elif self.act_angles[0] < -0.34 and self.act_angles[0] > -0.88:
        #         d = 1.59*self.dc
        # else:
        d = self.dc

        return d

    def JointStatesCallback(self, msg):
        self.act_angles = np.array(msg.data)

    def JointVelCallback(self, msg):
        self.act_vel = np.array(msg.data)

    def MoveServosProxy(self, req):
        self.act_load = req.torque

        # Enforce normalized actuator angles
        self.act_load = min(self.act_load,1)
        self.act_load = max(self.act_load,-1)

        return 0
        



if __name__ == '__main__':

    try:
        SimAcrobotNode()
    except rospy.ROSInterruptException:
        pass