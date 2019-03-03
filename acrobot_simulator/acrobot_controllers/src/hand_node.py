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
from std_msgs.msg import Float64, String
from hand_simulator.srv import MoveServos
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from std_srvs.srv import Empty, EmptyResponse

class SimHandNode():

    act_torque = np.array([0.,0.,0.,0.]) # left_proximal, left_distal, right_proximal, right_distal
    act_angles = None # actuator angles of all three fingers
    pinching_angle = 0
    fingers_angles = np.array([0.,0.,0.,0.]) # left_proximal, left_distal, right_proximal, right_distal
    fingers_vels = np.array([0.,0.,0.,0.]) # left_proximal, left_distal, right_proximal, right_distal
    ref_angles = np.array([0.,0.,0.,0.,0.,0.]) # spring reference angle left_proximal, left_distal, right_proximal, right_distal   
    lift_values = np.array([0.,0.])   
    lift_flag = False # False means lift down
    
    def __init__(self):
        rospy.init_node('SimHandNode', anonymous=True)

        if rospy.has_param('~gripper/gripper_type'):
            Gtype = rospy.get_param('~gripper/gripper_type')
            if Gtype=='reflex':
                self.num_fingers = 3
            if Gtype=='model_T42':
                self.num_fingers = 2
            if Gtype=='model_O':
                self.num_fingers = 3
            k1 = rospy.get_param('~' + Gtype + '/proximal_finger_spring_coefficient')
            k2 = rospy.get_param('~' + Gtype + '/distal_finger_spring_coefficient')
            d1 = rospy.get_param('~' + Gtype + '/proximal_finger_damping_coefficient')
            d2 = rospy.get_param('~' + Gtype + '/distal_finger_damping_coefficient')
            max_f = rospy.get_param('~' + Gtype + '/tendon_max_force') # max tendon force
            h = rospy.get_param('~' + Gtype + '/finger_tendon_force_distribution') # Tendon force distribution on the finger
            self.lift_values = rospy.get_param('~' + Gtype + '/lift_values')

        self.Q = np.array([[max_f, 0., 0.],[0., max_f, 0.],[0., 0., max_f]])
        self.R = np.array([[h[0],0.,0.],[h[1],0.,0.],[0.,h[0],0.],[0.,h[1],0.],[0.,0.,h[0]],[0.,0.,h[1]]])
        self.K = np.diag([k1,k2,k1,k2,k1,k2]) # Springs coefficients
        self.D = np.diag([d1,d2,d1,d2,d1,d2]) # Joints damping coefficients

        self.act_angles = np.zeros(self.num_fingers) # Normalized actuators angles [0,1]
        self.Q = self.Q[:self.num_fingers,:self.num_fingers]
        self.R = self.R[:2*self.num_fingers,:self.num_fingers]
        self.K = self.K[:2*self.num_fingers,:2*self.num_fingers]
        self.D = self.D[:2*self.num_fingers,:2*self.num_fingers]
        self.ref_angles = self.ref_angles[:2*self.num_fingers].reshape(2*self.num_fingers,1)

        #initialize service handlers:
        rospy.Service('MoveServos', MoveServos, self.MoveServosProxy)
        rospy.Service('LiftHand', Empty, self.LiftHandProxy)
        rospy.Service('PinchingAngle', MoveServos, self.PinchingAngleProxy)

        rospy.Subscriber('/hand/my_joint_states', Float32MultiArray, self.JointStatesCallback)
        rospy.Subscriber('/hand/my_joint_velocities', Float32MultiArray, self.JointVelCallback)

        self.pub_f1_jb1 = rospy.Publisher('/hand/base_to_finger_1_1_position_controller/command', Float64, queue_size=10)
        self.pub_f1_j12 = rospy.Publisher('/hand/finger_1_1_to_finger_1_2_position_controller/command', Float64, queue_size=10)
        self.pub_f1_j23 = rospy.Publisher('/hand/finger_1_2_to_finger_1_3_position_controller/command', Float64, queue_size=10)
        self.pub_f2_jb1 = rospy.Publisher('/hand/base_to_finger_2_1_position_controller/command', Float64, queue_size=10)
        self.pub_f2_j12 = rospy.Publisher('/hand/finger_2_1_to_finger_2_2_position_controller/command', Float64, queue_size=10)
        self.pub_f2_j23 = rospy.Publisher('/hand/finger_2_2_to_finger_2_3_position_controller/command', Float64, queue_size=10)
        self.pub_f3_jb2 = rospy.Publisher('/hand/base_to_finger_3_2_position_controller/command', Float64, queue_size=10)
        self.pub_f3_j23 = rospy.Publisher('/hand/finger_3_2_to_finger_3_3_position_controller/command', Float64, queue_size=10)
        self.pub_lift = rospy.Publisher('/hand/rail_to_base_controller/command', Float64, queue_size=10)

        self.gripper_angles_pub = rospy.Publisher('/gripper/pos', Float32MultiArray, queue_size=10)
        self.gripper_load_pub = rospy.Publisher('/gripper/load', Float32MultiArray, queue_size=10)
        self.gripper_lift_pub = rospy.Publisher('/gripper/lift_status', String, queue_size=10)
        msg = Float32MultiArray()
        msg_lift = String()

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():

            tendon_forces = self.Q.dot( self.act_angles.reshape(self.num_fingers,1) )

            if self.num_fingers == 3:
                self.act_torque = self.R.dot( tendon_forces ) - self.K.dot( self.fingers_angles - self.ref_angles ) # Damping was not implented here yet

                self.pub_f1_j12.publish(self.act_torque[0])
                self.pub_f1_j23.publish(self.act_torque[1])
                self.pub_f2_j12.publish(self.act_torque[2])
                self.pub_f2_j23.publish(self.act_torque[3])
                self.pub_f3_jb2.publish(self.act_torque[4])
                self.pub_f3_j23.publish(self.act_torque[5])
                
                self.pub_f1_jb1.publish(self.pinching_angle)
                self.pub_f2_jb1.publish(-self.pinching_angle)


            if self.num_fingers == 2:
                d = self.set_damping_matrix()

                self.act_torque = self.R.dot( tendon_forces ) - self.K.dot( self.fingers_angles - self.ref_angles ) - d.dot( self.fingers_vels ) # Damping does not work well because of numerical errors

                self.pub_f1_jb1.publish(self.act_torque[0])
                self.pub_f1_j12.publish(self.act_torque[1])
                self.pub_f2_jb1.publish(self.act_torque[2])
                self.pub_f2_j12.publish(self.act_torque[3])

            
            msg.data = self.act_angles
            self.gripper_angles_pub.publish(msg)
            msg.data = tendon_forces
            self.gripper_load_pub.publish(msg)
            

            self.pub_lift.publish(self.lift_values[int(self.lift_flag==True)])

            msg_lift.data = 'up' if self.lift_flag else 'down'
            self.gripper_lift_pub.publish(msg_lift)

            rate.sleep()

    def set_damping_matrix(self):
        d = np.zeros((2*self.num_fingers,2*self.num_fingers))
        for i in range(2*self.num_fingers):
            if np.abs(self.fingers_vels[i]) < 0.1e-2:
                d[i,i] = 0.0
                continue
            if np.abs(self.fingers_vels[i]) < 0.4e-2:
                d[i,i] = self.D[i,i]/4
                continue
            d[i,i] = self.D[i,i]

        return d

    def JointStatesCallback(self, msg):
        angles = np.array(msg.data)

        if self.num_fingers == 3:
            self.fingers_angles = angles[[1,2,4,5,6,7]].reshape(6,1)
            self.fingers_angles = self.fingers_angles[:2*self.num_fingers]
        else:
            self.fingers_angles = angles.reshape(4,1)

    def JointVelCallback(self, msg):
        vel = np.array(msg.data)

        if self.num_fingers == 3:
            self.fingers_vels = vel[[1,2,4,5,6,7]].reshape(6,1)
            self.fingers_vels = self.fingers_vels[:2*self.num_fingers]
        else:
            self.fingers_vels = vel.reshape(4,1)

    def MoveServosProxy(self, req):
        if (len(req.pos) < self.num_fingers):
            rospy.logerr("[hand_node] Command vector size (%d) is not compatible with the number (%d) of fingers." % (len(req.pos), self.num_fingers))
            return 1
        self.act_angles = np.array(req.pos[:self.num_fingers])

        # Enforce normalized actuator angles
        for i in range(self.num_fingers):
            self.act_angles[i] = min(self.act_angles[i],1)
            self.act_angles[i] = max(self.act_angles[i],0)

        return 0

    def LiftHandProxy(self, req):

        self.lift_flag = not self.lift_flag

        return EmptyResponse() 

    def PinchingAngleProxy(self, req):
        if len(req.pos) > 1:
            rospy.logerr("[hand_node] To many values in command vector.")

        self.pinching_angle = req.pos[0]

        return 0
        



if __name__ == '__main__':

    try:
        SimHandNode()
    except rospy.ROSInterruptException:
        pass