#!/usr/bin/python 

'''
----------------------------
Author: Avishai Sintov
        Rutgers University
Date: October 2018
----------------------------
'''


import rospy
import numpy as np 
from std_msgs.msg import Float64, Float32MultiArray
from gazebo_msgs.msg import LinkStates
from rosgraph_msgs.msg import Clock
import PyKDL

windowSize = 10

class my_joint_state_publisher():

    joint_angles = None#, dtype=np.float32)
    joint_angles_prev = None
    joint_vels = None
    order = np.array([0,0,0,0,0,0,0,0,0])
    msg = Float32MultiArray()
    Gtype = None
    current_time = 0.0
    prev_time = 0.0
    win = np.array([])

    def __init__(self):
        rospy.init_node('my_joint_state_publisher', anonymous=True)

        self.joint_angles = [0.,0.]
        self.joint_vels = [0.,0.]

        rospy.Subscriber('/gazebo/link_states', LinkStates, self.linkStatesCallback)
        rospy.Subscriber('/clock', Clock, self.ClockCallback)
        joint_states_pub = rospy.Publisher('/acrobot/my_joint_states', Float32MultiArray, queue_size=10)
        joint_vel_pub = rospy.Publisher('/acrobot/my_joint_velocities', Float32MultiArray, queue_size=10)

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            # self.UpdateVelocities() # Update is now done in linkStatesCallback(.)

            self.msg.data = self.joint_angles
            joint_states_pub.publish(self.msg)

            self.msg.data = self.joint_vels
            joint_vel_pub.publish(self.msg)   

            rate.sleep()

    def linkStatesCallback(self, msg):

        self.getNameOrder(msg.name)

        q_world = PyKDL.Rotation.Quaternion(msg.pose[self.order[0]].orientation.x, msg.pose[self.order[0]].orientation.y, msg.pose[self.order[0]].orientation.z, msg.pose[self.order[0]].orientation.w)
        q_1 = PyKDL.Rotation.Quaternion(msg.pose[self.order[1]].orientation.x, msg.pose[self.order[1]].orientation.y, msg.pose[self.order[1]].orientation.z, msg.pose[self.order[1]].orientation.w)
        q_2 = PyKDL.Rotation.Quaternion(msg.pose[self.order[2]].orientation.x, msg.pose[self.order[2]].orientation.y, msg.pose[self.order[2]].orientation.z, msg.pose[self.order[2]].orientation.w)
        dq_1 = msg.twist[self.order[1]].angular.y
        dq_2 = msg.twist[self.order[2]].angular.y
    
        if 1:
            a = (q_1.Inverse()*q_world).GetEulerZYX()
            if a[2] > 1:
                self.joint_angles[0] = a[1]
            elif a[1] > 0:
                self.joint_angles[0] = np.pi - a[1]
            else:
                self.joint_angles[0] = -(np.pi + a[1])
            b = (q_2.Inverse()*q_1).GetEulerZYX()
            if b[2] < 0.1:
                self.joint_angles[1] = -b[1]
            elif b[0] > 0:
                self.joint_angles[1] = -(np.pi - b[1])
            else:
                self.joint_angles[1] = np.pi + b[1]

            # For horizontal configuration
            # if a[2] > 0:
            #     self.joint_angles[0] = a[1]
            # elif a[0] < 0:
            #     self.joint_angles[0] = np.pi - a[1]
            # else:
            #     self.joint_angles[0] = -(np.pi + a[1])
            # b = (q_2.Inverse()*q_1).GetEulerZYX()
            # if b[0] > 3.13 or (b[0] < -3.13 and b[2] < -3.13):
            #     self.joint_angles[1] = b[1]
            # elif b[0] < 0:
            #     self.joint_angles[1] = np.pi - b[1]
            # else:
            #     self.joint_angles[1] = -(np.pi + b[1])
            # self.joint_angles[0] = -(np.pi - self.joint_angles[0]) if self.joint_angles[0] >= 0 else np.pi + self.joint_angles[0]
            # self.joint_angles[1] = self.joint_angles[1] - np.pi if self.joint_angles[1] >= 0 else self.joint_angles[1] + np.pi

            # Apply mean filter to joint velocities with windowSize
            self.joint_vels = np.array([dq_1, (dq_2 - dq_1)])
            # if self.win.shape[0] < windowSize:
            #     self.joint_vels = np.array([-dq_1, -(dq_2 - dq_1)]) # Currently finger velocity do not compensate base motion
            #     self.win = np.append(self.win, self.joint_vels).reshape(-1, 2)
            # else:
            #     v = np.array([-dq_1, -(dq_2 - dq_1)]) # Currently finger velocity do not compensate base motion
            #     self.win = np.append(self.win, v).reshape(-1, 2)
            #     self.joint_vels = np.mean(self.win, axis=0)
            #     self.win = np.delete(self.win, 0, axis=0)  
            self.joint_vels[np.abs(self.joint_vels) <= 0.1e-2] = 0.    

    def ClockCallback(self, msg):
        self.current_time = msg.clock.secs + msg.clock.nsecs * 1e-9

    def getNameOrder(self, names):

        for i, name in enumerate(names):
            # print(i,str)
            if name.find('ground_plane::link') > 0:
                self.order[0] = i
                continue
            if name.find('link1') > 0:
                self.order[1] = i
                continue
            if name.find('link2') > 0:
                self.order[2] = i
                continue

if __name__ == '__main__':

    try:
        my_joint_state_publisher()
    except rospy.ROSInterruptException:
        pass