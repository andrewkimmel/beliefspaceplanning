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
    gripper_load_prev = np.array([0., 0.])
    obj_pos = np.array([0., 0.])
    obj_vel = np.array([0., 0.])
    drop = True
    Done = False
    running = False
    action = np.array([0.,0.])
    state = np.array([0.,0., 0., 0.])
    joint_states = np.array([0., 0., 0., 0.])
    joint_velocities = np.array([0., 0., 0., 0.])
    n = 0
    
    texp = transition_experience(Load=True, discrete = discrete_actions, postfix='_bu')

    def __init__(self):
        rospy.init_node('actor_pub_record', anonymous=True)

        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/hand/obj_vel', Float32MultiArray, self.callbackObjVel)
        rospy.Subscriber('/hand/my_joint_states', Float32MultiArray, self.callbackJoints)
        rospy.Subscriber('/hand/my_joint_velocities', Float32MultiArray, self.callbackJointsVel)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)
        rospy.Subscriber('/collect/gripper_action', Float32MultiArray, self.callbackAction)

        rospy.Service('/actor/trigger', Empty, self.callbackTrigger)
        rospy.Service('/actor/save', Empty, self.callbackSave)

        rate = rospy.Rate(2)
        count = 0
        while not rospy.is_shutdown():

            if self.running:
                dL = self.gripper_load - self.gripper_load_prev
                self.gripper_load_prev = np.copy(self.gripper_load)
                self.state = np.concatenate((self.obj_pos, self.gripper_load, self.obj_vel, dL), axis=0)
                
                
                self.texp.add(self.state, self.action, self.state, self.drop)

                if self.drop:
                    print('[recorder] Episode ended (%d points so far).' % self.texp.getSize())
                    self.running = False
                    # if not (count % 20):
                        # self.texp.save()
                    # count += 1

            rate.sleep()

    def callbackGripperLoad(self, msg):
        if np.all(self.gripper_load_prev == 0) or np.any(np.abs(self.gripper_load_prev - self.gripper_load) > 0.5):
            self.gripper_load_prev = np.array(msg.data)
        self.gripper_load = np.array(msg.data)

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def callbackObjVel(self, msg):
        Obj_vel = np.array(msg.data)
        self.obj_vel = Obj_vel[:2] * 1000 # m/s to mm/s

    def callbackJoints(self, msg):
        self.joint_states = np.array(msg.data)

    def callbackJointsVel(self, msg):
        self.joint_velocities = np.array(msg.data)

    def callbackDrop(self, msg):
        self.drop = msg.data

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbackTrigger(self, msg):
        self.running = not self.running

        return EmptyResponse()

    def callbackSave(self, msg):
        self.texp.save()

        return EmptyResponse()


if __name__ == '__main__':
    
    try:
        actorPubRec()
    except rospy.ROSInterruptException:
        pass