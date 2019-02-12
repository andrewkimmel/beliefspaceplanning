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
import subprocess
import os

class collect_data():

    gripper_closed = False
    discrete_actions = True # Discrete or continuous actions
    drop = True

    num_episodes = 20000
    episode_length = 10000

    gz_steps = 300

    texp = transition_experience(Load=True, discrete = discrete_actions)
    

    # pauseMessage = pygazebo.msg.world_control_pb2.WorldControl()
    # pauseMessage.pause = 1

    # resumeMessage = pygazebo.msg.world_control_pb2.WorldControl()
    # resumeMessage.pause = 0

    # stepMessage = pygazebo.msg.world_control_pb2.WorldControl()
    # stepMessage.multi_step = gz_steps


    def __init__(self):
        rospy.init_node('collect_data', anonymous=True)

        rospy.Subscriber('/hand_control/gripper_status', String, self.callbackGripperStatus)
        self.pub_gripper_action = rospy.Publisher('/collect/gripper_action', Float32MultiArray, queue_size=10)
        rospy.Service('/collect/random_episode', Empty, self.run_random_episode)
        rospy.Service('/collect/planned_episode', rolloutReq, self.run_planned_episode)
        rospy.Service('/collect/process_data', Empty, self.process_data)
        rospy.Service('/collect/find_sparse_region', sparse_goal, self.find_sparse_region)
        self.obs_srv = rospy.ServiceProxy('/hand_control/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/hand_control/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/hand_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)
        self.rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)

        rospy.sleep(1.)

        print('[collect_data] Ready to collect...')

        self.rate = rospy.Rate(15) 
        # while not rospy.is_shutdown():
            # self.rate.sleep()
        rospy.spin()

    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"

    def callbackDrop(self, msg):
        self.drop = msg.data

    def process_data(self, msg):
        print('[collect_data] Proccessing data...')

        self.texp.save()
        # self.texp.process_transition_data(stepSize = 10, mode = 3, plot = False)
        # self.texp.process_svm(stepSize = 10, mode = 3)

        # print('[collect_data] Data processed and saved. Plotting current data...')
        # self.texp.plot_data()

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
        os.system("gz world -p 1") # pause (gz world -p 1)
        for ep_step in range(self.episode_length):

            if n == 0:
                action, n = self.choose_action()

            # msg.data = action
            # self.pub_gripper_action.publish(msg)
            suc = self.move_srv(action).success
            n -= 1

            os.system("gz world -m 200")
        
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
                os.system("gz world -p 0")
                break

            # self.rate.sleep()

        print('[collect_data] End of episode (%d points so far).'%(self.texp.getSize()))

        return EmptyResponse()

    def run_planned_episode(self, req):

        print('[collect_data] Rolling-out new planned actions...')
        state_seq = self.rollout_srv.call(req)

        self.texp.add_rollout_data() # Add rollout data to database

        if not state_seq.success:
            print('[collect_data] Rollout failed.')
            return state_seq

        msg = Float32MultiArray()
        Done = False

        print('[collect_data] Roll-out finished, running random actions...')

        # Start episode
        # subprocess.call(["gz", "world", "-p", "1"]) # pause (gz world -p 1)
        publisher.publish(pauseMessage)
        for ep_step in range(self.episode_length):

            if n == 0:
                action, n = self.choose_action()

            # msg.data = action
            # self.pub_gripper_action.publish(msg)
            suc = self.move_srv(action).success
            n -= 1

            subprocess.call(["gz world -m", str(self.gz_steps)])

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
                subprocess.call(["gz", "world", "-p", "0"])
                break

            # self.rate.sleep()

        print('[collect_data] End of episode (%d points so far).'%(self.texp.getSize()))

        return state_seq

    def choose_action(self, p = 0.5):
        if self.discrete_actions:
            A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])
            a = A[np.random.randint(A.shape[0])]
            if np.random.uniform(0,1,1) > p:
                if np.random.uniform(0,1,1) > 0.5:
                    a = A[0]
                else:
                    a = A[1]

            if np.random.uniform() > 0.5:
                if np.random.uniform() > 0.6:
                    num_steps = np.random.randint(200)
                else:
                    num_steps = np.random.randint(80)
            else:
                num_steps = np.random.randint(11,30)

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
