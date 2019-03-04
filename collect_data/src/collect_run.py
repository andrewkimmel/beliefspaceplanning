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
    episode_length = 2000

    texp = transition_experience(Load=True, discrete = discrete_actions)
    
    def __init__(self):
        rospy.init_node('collect_data', anonymous=True)

        self.pub_gripper_action = rospy.Publisher('/collect/gripper_action', Float32MultiArray, queue_size=10)
        rospy.Service('/collect/random_episode', Empty, self.run_random_episode)
        rospy.Service('/collect/planned_episode', rolloutReq, self.run_planned_episode)
        rospy.Service('/collect/save_data', Empty, self.save_data)
        self.obs_srv = rospy.ServiceProxy('/acrobot_control/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/acrobot_control/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/acrobot_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/acrobot_control/ResetGripper', Empty)
        self.record_srv = rospy.ServiceProxy('/actor/trigger', Empty)
        self.recorder_save_srv = rospy.ServiceProxy('/actor/save', Empty)
        
        rospy.sleep(1.)

        print('[collect_data] Ready to collect...')

        self.rate = rospy.Rate(2) 
        rospy.spin()

    def save_data(self, msg):
        print('[collect_data] Saving all data...')

        self.recorder_save_srv()
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
        action = np.array([0.])
        self.record_srv()
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
                fail = self.drop_srv().dropped # Check if dropped - end of episode
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

    def run_planned_episode(self, req):

        # Reset gripper
        self.reset_srv()

        print('[collect_data] Rolling-out new planned actions...')
        A = np.array(req.actions).reshape(-1, 2)

        msg = Float32MultiArray()

        state = np.array(self.obs_srv().state)

        # Start episode
        n = 0
        action = np.array([0.])
        self.record_srv()
        for ep_step in range(self.episode_length):

            if n == 0:
                if ep_step < A.shape[0]:
                    action = A[ep_step]
                    n = 1
                else:
                    action, n = self.choose_action()

            msg.data = action
            self.pub_gripper_action.publish(msg)
            suc = self.move_srv(action).success
            n -= 1

            # Get observation
            next_state = np.array(self.obs_srv().state)

            if suc:
                fail = self.drop_srv().dropped # Check if dropped - end of episode
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

        return {'states': [], 'actions_res': [], 'success': True}

    def choose_action(self):
        if self.discrete_actions:
            A = np.array([[1.0],[0.7],[0.3],[0.0],[-0.3],[-0.7],[-1.0]])
            a = A[np.random.randint(A.shape[0])]
            
            return a, 1
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
