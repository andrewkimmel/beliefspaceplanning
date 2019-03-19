#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gpup_gp_node.srv import one_transition
from control.srv import pathTrackReq

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class gp_controller():

    drop = True
    obj_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    action = np.array([0.,0.])
    exclude_action = np.array([0.,0.])
    goal = np.array([0.,0.,0.,0.])
    A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])

    def __init__(self):
        rospy.init_node('gp_controller', anonymous=True)

        self.gp = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)

        pub_best_action = rospy.Publisher('/gp_controller/action', Float32MultiArray, queue_size=10)
        # pub_2record = rospy.Publisher('/rollout/gripper_action', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/control/goal', Float32MultiArray, self.callbackGoal)
        rospy.Subscriber('/control/exclude', Float32MultiArray, self.callbackExclude)

        msg = Float32MultiArray()

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            state = np.concatenate((self.obj_pos, self.gripper_load), axis=0)

            self.action = self.check_all_action(state)

            msg.data = self.action
            pub_best_action.publish(msg)
            # pub_2record.publish(msg)
            rate.sleep()

    def check_all_action(self, s):
        goal = self.goal
        
        D = []
        # print "---------"
        for a in self.A:
            if np.all(a == self.exclude_action):
                D.append(1000)
                print "Action " + str(a) + " excluded."
                continue
            # print "Checking state ", s, " with action ", a, " size ", len(D)
            horizon = 15 if np.all(a == self.A[0]) or np.all(a == self.A[0]) else 1
            cur_s = np.copy(s)
            for i in range(horizon):
                res = self.gp(cur_s.reshape(-1,1), a)
                s_next = np.array(res.next_state)
                cur_s = np.copy(s_next)
                
            D.append(np.linalg.norm(goal-s_next))
        D = np.array(D)
        
        action = np.copy(self.A[np.argmin(D)])
        # if np.linalg.norm(goal[:2] - self.obj_pos) > 4.0:
        #     K = 0.4
        #     action *= K*np.linalg.norm(goal[:2] - self.obj_pos)

        return action

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)

    def callbackExclude(self, msg):
        self.exclude_action = np.array(msg.data)

if __name__ == '__main__':
    try:
        gp_controller()
    except rospy.ROSInterruptException:
        pass