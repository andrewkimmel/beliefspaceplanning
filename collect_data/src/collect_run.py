#!/usr/bin/env python

import rospy
import numpy as np
import time
import random
from std_msgs.msg import String, Float32MultiArray, Bool
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import rolloutReq, observation, IsDropped, TargetAngles
from sklearn.neighbors import NearestNeighbors 
from transition_experience import *
from collect_data.srv import sparse_goal
import matplotlib.pyplot as plt

class collect_data():

    gripper_closed = False
    discrete_actions = True # Discrete or continuous actions
    drop = True

    num_episodes = 20000
    episode_length = 10000

    texp = transition_experience(Load=True, discrete = discrete_actions)
    
    def __init__(self):
        rospy.init_node('collect_data', anonymous=True)

        self.pub_gripper_action = rospy.Publisher('/collect/gripper_action', Float32MultiArray, queue_size=10)
        rospy.Service('/collect/random_episode', Empty, self.run_random_episode)
        rospy.Service('/collect/save_data', Empty, self.save_data)

        self.obs_srv = rospy.ServiceProxy('/toy/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/toy/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/toy/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/toy/ResetGripper', Empty)

        rospy.Subscriber('/toy/cylinder_drop', Bool, self.callbackDrop)
        
        rospy.sleep(1.)

        print('[collect_data] Ready to collect...')

        self.rate = rospy.Rate(50) 
        rospy.spin()

    def callbackDrop(self, msg):
        self.drop = msg.data

    def save_data(self, msg):
        print('[collect_data] Saving all data...')

        self.texp.save()
        
        return EmptyResponse()

    def run_random_episode(self, req):

        # Reset gripper
        self.reset_srv()

        print('[collect_data] Running random episode...')

        Done = False
        msg = Float32MultiArray()

        state = np.array(self.obs_srv().state)

        # Start episode
        n = 0
        action = np.array([0.,0.])
        for ep_step in range(self.episode_length):

            if n == 0:
                action, n = self.choose_action()

            msg.data = action
            self.pub_gripper_action.publish(msg)
            suc = self.move_srv(action).success
            n -= 1

            # Get observation
            next_state = np.array(self.obs_srv().state)

            if suc:
                fail = self.drop # self.drop_srv().dropped # Check if dropped - end of episode
            else:
                # End episode if overload or angle limits reached
                rospy.logerr('[collect_data] Failed to move gripper. Episode declared failed.')
                fail = True

            self.texp.add(state, action, next_state, not suc or fail)
            state = np.copy(next_state)

            if not suc or fail:
                break

            self.rate.sleep()

        print('[collect_data] End of episode (%d points so far).'%(self.texp.getSize()))

        return EmptyResponse()

    def choose_action(self):
        if self.discrete_actions:
            A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])
            a = A[np.random.randint(A.shape[0])]

            if np.random.uniform() > 0.5:
                num_steps = np.random.randint(30)
            else:
                num_steps = 1

            return a, num_steps
        else:
            a = np.random.uniform(-1.,1.,2)
            if np.random.uniform(0,1,1) > 0.35:
                if np.random.uniform(0,1,1) > 0.5:
                    a[0] = np.random.uniform(-1.,-0.8,1)
                    a[1] = np.random.uniform(-1.,-0.8,1)
                else:
                    a[0] = np.random.uniform(0.8,1.,1)
                    a[1] = np.random.uniform(0.8,1.,1)

            return a

if __name__ == '__main__':
    try:
        collect_data()
    except rospy.ROSInterruptException:
        pass
