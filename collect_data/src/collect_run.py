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

        rospy.Subscriber('/hand_control/gripper_status', String, self.callbackGripperStatus)
        self.pub_gripper_action = rospy.Publisher('/collect/gripper_action', Float32MultiArray, queue_size=10)
        rospy.Service('/collect/random_episode', Empty, self.run_random_episode)
        rospy.Service('/collect/planned_episode', rolloutReq, self.run_planned_episode)
        rospy.Service('/collect/save_data', Empty, self.save_data)
        rospy.Service('/collect/find_sparse_region', sparse_goal, self.find_sparse_region)
        self.obs_srv = rospy.ServiceProxy('/hand_control/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/hand_control/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/hand_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)
        #self.rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)
        self.record_srv = rospy.ServiceProxy('/actor/trigger', Empty)
        self.recorder_save_srv = rospy.ServiceProxy('/actor/save', Empty)
        
        rospy.sleep(1.)

        print('[collect_data] Ready to collect...')

        self.rate = rospy.Rate(2) 
        # while not rospy.is_shutdown():
            # self.rate.sleep()
        rospy.spin()

    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"

    def callbackDrop(self, msg):
        self.drop = msg.data

    def save_data(self, msg):
        print('[collect_data] Saving all data...')

        self.recorder_save_srv()
        self.texp.save()
        
        return EmptyResponse()

    def run_random_episode(self, req):

        # Reset gripper
        self.reset_srv()
        while not self.gripper_closed:
            self.rate.sleep()

        print('[collect_data] Running random episode...')

        Done = False
        msg = Float32MultiArray()

        state = np.array(self.obs_srv().state)

        # Start episode
        n = 0
        action = np.array([0.,0.])
        self.record_srv()
        for ep_step in range(self.episode_length):

            if n == 0:
                action, n = self.choose_action(0.9)

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

    def run_planned_episode(self, req):

        # Reset gripper
        self.reset_srv()
        while not self.gripper_closed:
            self.rate.sleep()

        print('[collect_data] Rolling-out new planned actions...')
        #state_seq = self.rollout_srv.call(req)
        A = np.array(req.actions).reshape(-1, 2)

        #self.texp.add_rollout_data() # Add rollout data to database

        #if not state_seq.success:
        #    print('[collect_data] Rollout failed.')
        #    return state_seq

        msg = Float32MultiArray()

        #print('[collect_data] Roll-out finished, running random actions...')

        state = np.array(self.obs_srv().state)

        # Start episode
        n = 0
        action = np.array([0.,0.])
        self.record_srv()
        for ep_step in range(self.episode_length):

            if n == 0:
                if ep_step < A.shape[0]:
                    action = A[ep_step]
                    n = 1
                else:
                    action, n = self.choose_action(0.7)

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

        return {'states': [], 'actions_res': [], 'success': True}

    def choose_action(self, p = 0.5):
        if self.discrete_actions:
            A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])
            a = A[np.random.randint(A.shape[0])]
            if np.random.uniform(0,1,1) > p:
                if np.random.uniform(0,1,1) > 0.5:
                    a = A[0]
                else:
                    a = A[1]

            if np.random.uniform() > 0.8:
                if np.random.uniform() > 0.5:
                    num_steps = np.random.randint(400)
                else:
                    num_steps = np.random.randint(160)
            else:
                num_steps = np.random.randint(15,65)

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

    def find_sparse_region(self, req):

        radius = 7
        xy = np.array([item[0] for item in self.texp.memory])[:,:2]
        neigh = NearestNeighbors(radius=radius).fit(xy)
        
        sample = np.array([0.,0.])
        min_nn = 1e10
        goal = np.copy(sample)
        for _ in range(200):
            sample[0] = np.random.uniform(-100.,100.)
            sample[1] = np.random.uniform(40.,140.)
            rng = neigh.radius_neighbors([sample])
            if len(rng[0][0]) <= 500:
                continue
            if len(rng[0][0]) < min_nn:
                min_nn = len(rng[0][0])
                goal = np.copy(sample)
        # sample = np.array([0.,0.])
        # sample[0] = np.random.uniform(-100.,100.)
        # sample[1] = np.random.uniform(40.,140.)
        # goal = np.copy(sample)
        
        return {'goal': goal}


if __name__ == '__main__':
    
    try:
        collect_data()
    except rospy.ROSInterruptException:
        pass
