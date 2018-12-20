#!/usr/bin/env python

import numpy as np
import time
import random
import matplotlib.pyplot as plt


class SquareEnv():
    visited_states = np.array([[0.,0.]])
    
    # noise_std = 0.00001
    noise_std = 0.06

    def __init__(self, size=1, reward_type='sparse', thr = .4, add_noise = True):
        self.size = size
        self.reward_type = reward_type
        self.thr = thr # distance threshold
        self.add_noise = add_noise
        self.state = np.array([0.,0.])

    def reset(self):
        # self.state = np.random.uniform(-self.size, self.size, 2) # Start from a random position
        self.state = np.array([-0.95,-0.95])# + np.random.normal(0, 0.05) # Always start from [0,0] with some uncertainty
        return np.copy(self.state)

    def step(self, action, scale=5): # Action is -1 to 1 in each dimension
        self.state[0] += action[0]/scale
        self.state[1] += action[1]/scale
        if self.add_noise: 
            self.state += np.random.normal(0., self.noise(self.state), 2) # Add noise

        return np.copy(self.state)

    def Step(self, state, action, scale=5): # Action is -1 to 1 in each dimension
        next_state = np.array([0.,0.])
        next_state[0] = state[0] + action[0]/scale
        next_state[1] = state[1] + action[1]/scale

        noise = self.noise(state)
        if self.add_noise: 
            next_state += np.random.normal(0., self.noise(self.state), 2) # Add noise

        return np.copy(next_state), np.array([self.noise(self.state),self. noise(self.state)])

    def noise(self, state):
            return 0.005

    def render(self):
        print("state :" + np.array_str(self.state) + ", goal :" + np.array_str(self.goal) + ", distance to goal: " + str(np.linalg.norm(self.state-self.goal)))

    def log(self, reward, done):
        self.visited_states = np.append(self.visited_states, [self.state], axis=0)

    def plot(self):
        plt.figure(9)
        plt.plot(self.visited_states[1:,0], self.visited_states[1:,1],'.b', label='visited states')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title(self.title)
        # plt.show()


class ProbEnv():
    def __init__(self, size=1, thr = 0.7, add_noise = True):
        self.size = size
        self.thr = thr
        self.add_noise = add_noise

    def probability(self, state):

        x = state[0]
        y = state[1]

        if x <= -0.75 or x >= 0.75 or y <= -0.75:
            p = 0.02
        else:
            p = 0.2

        # if self.add_noise:
            # p += np.random.normal(0., 0.0005) # Add uncertainty

        p = 1. if p > 1 else p
        p = 0. if p < 0 else p

        return p, p > self.thr

 



