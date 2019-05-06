#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray, Float32
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pickle
from acrobot_control.srv import pathTrackReq
from prx_simulation.srv import simulation_observation_srv
from math import sin, cos
import sys

''' 
git clone https://github.com/python-control/python-control.git
cd python-control/
python setup.py install
pip install slycot
'''
sys.path.insert(0, '/home/pracsys/Documents/python-control')
from control import lqr

sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class vs_controller():

    drop = True
    state = np.array([0., 0., 0., 0.])
    action = 0.0
    goal = np.array([0., 0., 0., 0.])
    controller = 'lqr' # 'lqr' or 'pd'
    max_torque = 7.0
    planned_action = 0.0

    def __init__(self):
        rospy.init_node('vs_controller', anonymous=True)

        # rospy.Subscriber('/acrobot/state', Float32MultiArray, self.callbackState)
        get_obs_srv = rospy.ServiceProxy('/getObservation', simulation_observation_srv)
        pub_best_action = rospy.Publisher('/controller/action', Float32, queue_size=10)
        rospy.Subscriber('/control/goal', Float32MultiArray, self.callbackGoal)
        rospy.Subscriber('/control/planned_action', Float32, self.callbackPlannedAction)

        msg = Float32()

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.state = get_obs_srv().state
            self.state = [(0.0 if np.abs(s) < 1e-6 else s) for s in self.state]

            self.action = self.get_action()

            msg.data = self.action
            pub_best_action.publish(msg)
            rate.sleep()

    def get_action(self):

        e = self.wrapEuclidean(self.state, self.goal)
        e = [0.0 if np.abs(ee) < 1e-4 else ee for ee in e]

        if self.controller == 'pd':
            K = np.array([1000., 500., 110.0, 120.0])
            action = -np.dot( K, e)
            # print action, e, self.state, self.goal
            return action.item()
        
        if self.controller == 'lqr':
            """Solve the continuous time lqr controller.
            dx/dt = A x + B u
            cost = integral x.T*Q*x + u.T*R*u
            """
            Q = np.diag([100., 100.0, 1.0, 1.0])*100.
            R = 1.0*np.diag([1.])
            invR = 1./R # scipy.linalg.inv(R)
            A, B, ueq = self.linear_acrobot_system(self.state)

            try:
                # X = np.matrix(scipy.linalg.solve_continuous_are(A, B.reshape(-1,1), Q, R))
                # K = np.matrix(invR*(B*X).reshape(-1,1))
                # K = np.squeeze(np.asarray(K))
                K, _, _ = lqr(A, np.matrix(B).reshape(-1,1), Q, R)
                action = -np.dot(K, e)[0] + ueq
                # action = self.max_torque if action > self.max_torque else action.item()
                # action = -self.max_torque if action < -self.max_torque else action
            except:
                if np.all(np.abs(self.state)<1e-8):
                    action = 1e-1
                else:
                    action = 0.0
            return action

    # def callbackState(self, msg):
    #     self.state = np.array(msg.data)

    def linear_acrobot_system(self, x):
        lc = .5
        lc2 = .25
        l2 = 1.0
        m = 1.0
        I1 = 0.2
        I2 = 1.0
        l = 1.0
        g = 9.8

        # M = np.zeros((2,2))
        # M[1,1] = m * lc2 + m * (l2 + lc2 + 2 * l * lc * cos(theta2)) + I1 + I2
        # M[2,2] = m * lc2 + I2
        # M[1,2] = m * (lc2 + l * lc * cos(theta2)) + I2
        # M[2,1] = M[1,2]
        # C = np.zeros((2,1))
        # C[1] = -m * l * lc * theta2dot * theta2dot * sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * sin(theta2))
        # C[2] = m * l * lc * theta1dot * theta1dot * sin(theta2)
        # G = np.zeros((2,1))
        # G[1] = (m * lc + m * l) * g * cos(theta1) + (m * lc * g * cos(theta1 + theta2))
        # G[2] = m * lc * g * cos(theta1 + theta2)

        theta1 = x[0] - np.pi/2
        theta2 = x[1] 
        theta1dot = x[2]
        theta2dot = x[3]

        u = self.planned_action
        # g*lc*m*cos(theta1 + theta2)

        A = np.array([[ 0, 0, 1, 0],[ 0, 0, 0, 1],
            [((g*sin(theta1)*(l*m + lc*m) + g*lc*m*sin(theta1 + theta2))*(I2 + lc2*m) - g*lc*m*sin(theta1 + theta2)*(I2 + m*(lc2 + l*lc*cos(theta2))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2),                                                                                                                                                                                                                                                                  - ((- l*lc*m*cos(theta2)*theta1dot**2 + g*lc*m*sin(theta1 + theta2))*(I2 + m*(lc2 + l*lc*cos(theta2))) - (I2 + lc2*m)*(l*lc*m*cos(theta2)*theta2dot**2 + 2*l*lc*m*theta1dot*cos(theta2)*theta2dot + g*lc*m*sin(theta1 + theta2)) + l*lc*m*sin(theta2)*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2)))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2) - ((2*l*lc*m*sin(theta2)*(I2 + lc2*m) - 2*l*lc*m*sin(theta2)*(I2 + m*(lc2 + l*lc*cos(theta2))))*((I2 + lc2*m)*(- l*lc*m*sin(theta2)*theta2dot**2 - 2*l*lc*m*theta1dot*sin(theta2)*theta2dot + theta1dot/10 + g*cos(theta1)*(l*m + lc*m) + g*lc*m*cos(theta1 + theta2)) - (I2 + m*(lc2 + l*lc*cos(theta2)))*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2)**2,                                           ((2*l*lc*m*theta2dot*sin(theta2) - 1/10)*(I2 + lc2*m) + 2*l*lc*m*theta1dot*sin(theta2)*(I2 + m*(lc2 + l*lc*cos(theta2))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2),                                                   (I2/10 + (m*(lc2 + l*lc*cos(theta2)))/10 + 2*l*lc*m*sin(theta2)*(theta1dot + theta2dot)*(I2 + lc2*m))/(I1*I2 + lc2**2*m**2 + l2*lc2*m**2 + I2*l2*m + I1*lc2*m + I2*lc2*m - l**2*lc**2*m**2*cos(theta2)**2)],
            [ -((g*sin(theta1)*(l*m + lc*m) + g*lc*m*sin(theta1 + theta2))*(I2 + m*(lc2 + l*lc*cos(theta2))) - g*lc*m*sin(theta1 + theta2)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2), - ((I2 + m*(lc2 + l*lc*cos(theta2)))*(l*lc*m*cos(theta2)*theta2dot**2 + 2*l*lc*m*theta1dot*cos(theta2)*theta2dot + g*lc*m*sin(theta1 + theta2)) - (- l*lc*m*cos(theta2)*theta1dot**2 + g*lc*m*sin(theta1 + theta2))*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - 2*l*lc*m*sin(theta2)*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2)) + l*lc*m*sin(theta2)*(- l*lc*m*sin(theta2)*theta2dot**2 - 2*l*lc*m*theta1dot*sin(theta2)*theta2dot + theta1dot/10 + g*cos(theta1)*(l*m + lc*m) + g*lc*m*cos(theta1 + theta2)))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2) - (((I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m)*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2)) - (I2 + m*(lc2 + l*lc*cos(theta2)))*(- l*lc*m*sin(theta2)*theta2dot**2 - 2*l*lc*m*theta1dot*sin(theta2)*theta2dot + theta1dot/10 + g*cos(theta1)*(l*m + lc*m) + g*lc*m*cos(theta1 + theta2)))*(2*l*lc*m*sin(theta2)*(I2 + lc2*m) - 2*l*lc*m*sin(theta2)*(I2 + m*(lc2 + l*lc*cos(theta2)))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2)**2, -((I2 + m*(lc2 + l*lc*cos(theta2)))*(2*l*lc*m*theta2dot*sin(theta2) - 1/10) + 2*l*lc*m*theta1dot*sin(theta2)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2), -(I1/10 + I2/10 + (m*(l2 + lc2 + 2*l*lc*cos(theta2)))/10 + (lc2*m)/10 + 2*l*lc*m*sin(theta2)*(theta1dot + theta2dot)*(I2 + lc2*m + l*lc*m*cos(theta2)))/(I1*I2 + lc2**2*m**2 + l2*lc2*m**2 + I2*l2*m + I1*lc2*m + I2*lc2*m - l**2*lc**2*m**2*cos(theta2)**2)]])
 
        B = np.array([0, 0, -(I2 + m*(lc2 + l*lc*cos(theta2)))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2), (I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m)/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2)])

        return A, B, u
 
    def wrapEuclidean(self, x, y):
        v = np.pi

        d = 0
        s = 0
        e = []
        for i in range(len(x)):
            d = x[i]-y[i]
            if i < 2 and np.abs(d) > v:
                d = 2*v - np.abs(d)
                if x[i] > 0.0 and y[i] < 0.0:
                    d *= -1.0                
            e.append(d)

        return np.array(e)

    def callbackGoal(self, msg):
        self.goal = np.array(msg.data)

    def callbackPlannedAction(self, msg):
        self.planned_action = msg.data

if __name__ == '__main__':
    try:
        vs_controller()
    except rospy.ROSInterruptException:
        pass