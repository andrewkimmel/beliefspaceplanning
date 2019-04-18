#!/usr/bin/env python

import rospy
import numpy as np
import time
import random
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import gets

state_form = 'pos_load_vel' # 'pos_load' or 'pos_vel' or 'pos_load_vel' or 'pos_load_joints', or 'pos_joints'

class rolloutRec():
    discrete_actions = True

    gripper_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    gripper_load_prev = np.array([0., 0.])
    obj_pos = np.array([0., 0.])
    obj_vel = np.array([0., 0.])
    drop = True
    Done = False
    running = False
    action = np.array([0.,0.])
    state = np.array([0.,0., 0., 0.])
    joint_states = np.array([0., 0., 0., 0.])
    D_load = np.array([0., 0.])
    n = 0
    S = []
    A = []
    
    def __init__(self):
                
        rospy.init_node('rollout_recorder', anonymous=True)

        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/hand/obj_vel', Float32MultiArray, self.callbackObjVel)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)
        rospy.Subscriber('/rollout/gripper_action', Float32MultiArray, self.callbackAction)
        rospy.Subscriber('/hand/my_joint_states', Float32MultiArray, self.callbackJoints)

        rospy.Service('/rollout_recorder/trigger', Empty, self.callbackTrigger)
        rospy.Service('/rollout_recorder/get_states', gets, self.get_states)

        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.D_load = np.copy(self.gripper_load) - np.copy(self.gripper_load_prev)
            self.gripper_load_prev = np.copy(self.gripper_load)

            if self.running:
                if state_form == 'pos_load':
                    self.state = np.concatenate((self.obj_pos, self.gripper_load), axis=0)
                elif state_form == 'pos_load_vel':   
                    self.state = np.concatenate((self.obj_pos, self.gripper_load, self.obj_vel, self.D_load), axis=0)
                elif state_form == 'pos_load_joints':   
                    self.state = np.concatenate((self.obj_pos, self.gripper_load, self.joint_states), axis=0)
                elif state_form == 'pos_joints':   
                    self.state = np.concatenate((self.obj_pos, self.joint_states), axis=0)

                self.S.append(self.state)
                self.A.append(self.action)
                
                if self.drop:
                    print('[rollout_recorder] Episode ended.')
                    self.running = False

            rate.sleep()

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def callbackObjVel(self, msg):
        Obj_vel = np.array(msg.data)
        self.obj_vel = Obj_vel[:2] * 1000 # m/s to mm/s
    
    def callbackJoints(self, msg):
        self.joint_states = np.array(msg.data)

    def callbackDrop(self, msg):
        self.drop = msg.data

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbackTrigger(self, msg):
        self.running = True
        self.S = []
        self.A = []

        return EmptyResponse()

    def get_states(self, msg):

        return {'states': np.array(self.S).reshape((-1,)), 'actions': np.array(self.A).reshape((-1,))}

       
if __name__ == '__main__':
    
    try:
        rolloutRec()
    except rospy.ROSInterruptException:
        pass