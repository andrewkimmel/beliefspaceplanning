#!/usr/bin/env python

import numpy as np
import time
import random
import matplotlib.pyplot as plt


class ChaseEnv():
    visited_states = np.array([[0.,0.]])
    reached_goals = np.array([[0.,0.]])
    failed_goals = np.array([[0.,0.]])

    def __init__(self, size=1, reward_type='sparse', yfactor=10, thr = .4, add_noise = True):
        self.size = size
        self.reward_type = reward_type
        self.thr = thr # distance threshold
        self.yfactor = yfactor
        self.add_noise = add_noise
        self.state = np.array([0.,0.])

    def reset(self):
        self.state = np.random.uniform(-self.size, self.size, 2)
        return np.copy(self.state)

    def step(self, action, scale=10): # Action is -1 to 1 in each dimension
        self.state[0] += action[0]/scale
        self.state[1] += action[1]/scale/self.yfactor
        if self.add_noise: 
            self.state += np.random.normal(0., 0.02, 2) # Add noise

        return np.copy(self.state)

    def render(self):
        print("state :" + np.array_str(self.state) + ", goal :" + np.array_str(self.goal) + ", distance to goal: " + str(np.linalg.norm(self.state-self.goal)))

    def log(self, reward, done):
        self.visited_states = np.append(self.visited_states, [self.state], axis=0)
        if done and reward == 0:
            self.reached_goals = np.append(self.reached_goals, [self.goal], axis=0)
        elif done:
            self.failed_goals = np.append(self.failed_goals, [self.goal], axis=0)

    def plot(self):
        plt.figure(9)
        plt.plot(self.failed_goals[1:,0], self.failed_goals[1:,1],'.r', label='failed goals')
        plt.plot(self.reached_goals[1:,0], self.reached_goals[1:,1],'.g', label='reached goals')
        plt.plot(self.visited_states[1:,0], self.visited_states[1:,1],'.b', label='visited states')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title(self.title)
        # plt.show()


class ProbEnv():
    def __init__(self, size=1, yfactor=10, thr = 0.7, add_noise = True):
        self.size = size
        self.yfactor = yfactor/5
        self.thr = thr
        self.add_noise = add_noise

    def probability(self, state):

        x = state[0]
        y = state[1]

        p = self.prob_func(x,y) / self.prob_func(self.size, self.size)

        if self.add_noise:
            p += np.random.normal(0., 0.1) # Add uncertainty

        return p, True if p > self.thr else False # Returns the probability of failure and if it is above the threshold

    def prob_func(self, x, y):
        d1 = x - y
        d2 = x + y

        if np.sqrt(x**2 + y**2) < 0.45:
            return 0

        if (d1 >= 0 and d2 >= 0) or (d1 < 0 and d2 < 0):
            p = x**2
        if (d1 >= 0 and d2 < 0) or (d1 < 0 and d2 >= 0):
            p = self.yfactor * y**2

        return p






