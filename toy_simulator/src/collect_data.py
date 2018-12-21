#!/usr/bin/env python

import rospy
import numpy as np
import time
import random
import pickle
from std_msgs.msg import String, Float32MultiArray
from std_srvs.srv import Empty, EmptyResponse
from toy_simulator.srv import TargetAngles, IsDropped, observation, transition
import os.path
import matplotlib.pyplot as plt

SIZE = 1.

class transition_experience():
    path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/toy_simulator/data/'

    def __init__(self, Load=True, discrete = False):

        if discrete:
            self.mode = 'discrete'
        else:
            self.mode = 'cont'
        
        self.file_name = self.path + 'transition_data_' + self.mode + '.obj'

        if Load:
            self.load()
        else:
            self.clear()
        
    def add(self, state, action, next_state, done):
        self.memory += [(state, action, next_state, done)]
        
    def clear(self):
        self.memory = []

    def load(self):
        if os.path.isfile(self.file_name):
            print('Loading data from ' + self.file_name)
            with open(self.file_name, 'rb') as filehandler:
            # filehandler = open(self.file_name, 'r')
                self.memory = pickle.load(filehandler)
            print('Loaded transition data of size %d.'%self.getSize())
        else:
            self.clear()

    def getComponents(self):

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])

        return states, actions, next_states


    def save(self):
        print('Saving data...')
        file_pi = open(self.file_name, 'wb')
        pickle.dump(self.memory, file_pi)
        print('Saved transition data of size %d.'%self.getSize())
        file_pi.close()

    def getSize(self):
        return len(self.memory)

    def plot_data(self):

        states = [item[0] for item in self.memory]
        done = [item[3] for item in self.memory]
        states = np.array(states)
        failed_states = states[done]

        plt.figure(1)
        plt.plot(states[:,0],states[:,1],'-k')
        plt.plot(states[:,0],states[:,1],'.y')
        plt.plot(failed_states[:,0],failed_states[:,1],'.r')
        # plt.set(title='Object position')
        
        plt.show()

    def save_to_file(self):

        filen = self.path + 'transition_data_' + self.mode + '.db'

        n = self.getSize()

        states = np.array([item[0] for item in self.memory])
        actions = np.array([item[1] for item in self.memory])
        next_states = np.array([item[2] for item in self.memory])
        done = np.array([item[3] for item in self.memory])

        inx = np.where(done)

        # for i in range(len(done)):
        #     if done[i]:
        #         next_states[i] = np.array([-1000.,-1000.,-1000.,-1000.])

        M = np.concatenate((states, actions, next_states), axis=1)
        M = np.delete(M, inx, 0)

        np.savetxt(filen, M, delimiter=' ')


class collect_data():
    stCollecting = True # Enable collection
    discrete_actions = False # Discrete or continuous actions

    num_episodes = 20000
    episode_length = 10000

    texp = transition_experience(discrete=discrete_actions)

    def __init__(self):
        rospy.init_node('collect_data', anonymous=True)

        obs_srv = rospy.ServiceProxy('/toy/observation', observation)
        drop_srv = rospy.ServiceProxy('/toy/IsObjDropped', IsDropped)
        move_srv = rospy.ServiceProxy('/toy/MoveGripper', TargetAngles)
        reset_srv = rospy.ServiceProxy('/toy/ResetGripper', Empty)
        trans_srv = rospy.ServiceProxy('/toy/transition', transition)
        rospy.Service('/RL/start_collecting', Empty, self.start_collecting)

        msg = Float32MultiArray()

        rate = rospy.Rate(15) # 15hz
        while not rospy.is_shutdown():
            
            if self.stCollecting:

                for n in range(self.num_episodes):

                    # Reset gripper
                    reset_srv()

                    print('[collect_data] Episode %d (%d points so far).'%(n, self.texp.getSize()))

                    Done = False

                    # Start episode
                    for ep_step in range(self.episode_length):

                        # Get observation and choose action
                        state = np.array(obs_srv().state)
                        action = self.choose_action()
                        
                        msg.data = action
                        suc = move_srv(action)
                        rospy.sleep(0.01)

                        # Get observation
                        next_state = np.array(obs_srv().state)

                        if suc:
                            fail = drop_srv().dropped # Check if dropped - end of episode
                        else:
                            # End episode if overload or angle limits reached
                            rospy.logerr('[RL] Failed to move gripper. Episode declared failed.')
                            fail = True

                        self.texp.add(state, action, next_state, not suc or fail)
                        state = next_state

                        if not suc or fail:
                            Done = True
                            break

                        rate.sleep()

                    if n == self.num_episodes-1 or self.texp.getSize() >= 50000:
                        self.stCollecting = False
                        print('Finished running %d episodes!' % n)
                        self.texp.plot_data()
                        self.texp.save_to_file()
                        break

                    if n % 100 == 0:
                        self.texp.save()

                    if not self.stCollecting:
                        break
                
                self.texp.save()             

            self.texp.plot_data()
            rate.sleep()

    def start_collecting(self, msg):
        self.stCollecting = not self.stCollecting

    def choose_action(self):
        if self.discrete_actions:
            A = np.array([[1.,1.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,0.],[-1.,0.],[0.,-1.],[0.,1.]])
            a = A[np.random.randint(A.shape[0])]
        else:
            a = np.random.uniform(-1.,1.,2)

        return a



if __name__ == '__main__':
    
    try:
        collect_data()
    except rospy.ROSInterruptException:
        pass




