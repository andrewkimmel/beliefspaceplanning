#!/usr/bin/env python

import numpy as np
import time
import random
import matplotlib.pyplot as plt
import varz as V

class SquareEnv():

    visited_states = np.array([0.,0.])

    def __init__(self):
        self.size = V.SIZE
        self.state = np.array([0.,0.])

    def reset(self):
        # self.state = np.random.uniform(-self.size, self.size, 2) # Start from a random position
        self.state = np.array([-0.95,-0.95]) + np.random.normal(0, V.START_STD, 2) 
        self.fail = False

        # self.visited_states = np.copy(self.state.reshape(1,2))
        return np.copy(self.state)

    def step(self, action): # Action is -1 to 1 in each dimension
        self.state[0] += action[0]/V.SCALE
        self.state[1] += action[1]/V.SCALE

        if self.state[0] > V.SIZE or self.state[0] < -V.SIZE or self.state[1] > V.SIZE or self.state[1] < -V.SIZE:
            self.fail = True
        else:
            std = self.stdDet(self.state)
            suc = self.successDet(self.state)

            self.state += np.random.normal(0., std, 2) # Add noise

            if np.random.uniform() > suc:
                self.fail = True
            else:
                self.fail = False

        return np.copy(self.state), self.fail

    def stdDet(self, state):
        if state[0] < -V.B or state[0] > V.B:
            return 0.0

        if state[1] < 0.:
            return V.HIGH_STD
        else:
            return V.LOW_STD

    def successDet(self, state):
        if state[0] < -V.B or state[0] > V.B:
            return 1.0

        if state[1] < -0.5 or (state[1] > 0. and state[1] < 0.5):
            return V.LOW_SUC
        else:
            return V.HIGH_SUC

    def render(self):
        print("state :" + np.array_str(self.state) + ', fail: ' + str(self.fail))

    def log(self):
        self.visited_states = np.append(self.visited_states, np.copy(self.state.reshape(1,2)), axis=0)

    def plot(self):
        plt.figure(9)
        plt.plot(self.visited_states[:,0], self.visited_states[:,1],'.b', label='visited states')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title(self.title)
        plt.xlim([-V.SIZE, V.SIZE])
        plt.ylim([-V.SIZE, V.SIZE])
        plt.show()



 



