#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray, Float32
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gpup_gp_node.srv import one_transition
from acrobot_control.srv import pathTrackReq
from rollout_node.srv import gets
from prx_simulation.srv import simulation_observation_srv, simulation_valid_srv, simulation_reset_srv, simulation_action_srv
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

class general_control():

    drop = True
    state = np.array([0., 0., 0., 0.])
    action = 0.
    tol = 0.02
    goal_tol = 0.3
    horizon = 1
    goal = np.array([0., 0., 0., 0.])

    def __init__(self):
        rospy.init_node('control', anonymous=True)

        rospy.Service('/control', pathTrackReq, self.CallbackTrack)
        # rospy.Subscriber('/controller/action', Float32, self.CallbackBestAction)
        # self.pub_current_goal = rospy.Publisher('/control/goal', Float32MultiArray, queue_size=10)
        # self.pub_planned_action = rospy.Publisher('/control/planned_action', Float32, queue_size=10)

        self.plot_clear_srv = rospy.ServiceProxy('/plot/clear', Empty)
        self.plot_ref_srv = rospy.ServiceProxy('/plot/ref', pathTrackReq)

        self.obs_srv = rospy.ServiceProxy('/getObservation', simulation_observation_srv)
        self.reset_srv = rospy.ServiceProxy('/reset', simulation_reset_srv)
        self.action_srv = rospy.ServiceProxy('/sendAction', simulation_action_srv)
        self.valid_srv = rospy.ServiceProxy('/isValid', simulation_valid_srv)

        self.state_dim = var.state_dim_
        self.action_dim = var.state_action_dim_-var.state_dim_
        self.stepSize = var.stepSize_

        self.controller = 'lqr'

        print("[control] Ready to track path...")
        self.rate = rospy.Rate(80) 
        rospy.spin()

    def CallbackTrack(self, req):

        path = np.array(req.desired_path).reshape(-1, self.state_dim)
        Apath = np.array(req.action_path)
        self.openOrClosed = req.closed
        
        real_path, actions, success = self.run_tracking(path, Apath)

        return {'real_path': real_path, 'actions': actions, 'success' : success}

    def weightedL2(self, s1, s2, W = np.diag(np.array([1.,1.,0.2, 0.2]))):
        # return np.sqrt( np.dot(ds.T, np.dot(W, ds)) )
        return self.wrapEuclideanNorm(s2[:2], s2[:2])

    def run_tracking(self, S, A):

        # Reset Acrobot
        self.reset_srv()

        i_path = 1
        # msg = Float32MultiArray()
        # msg.data = S[i_path,:]
        # msge = Float32MultiArray()
        # for i in range(2):
        #     self.pub_current_goal.publish(msg)
        #     self.rate.sleep()
        # rospy.sleep(1.0)
        
        # self.plot_clear_srv()
        # self.plot_ref_srv(S.reshape((-1,)))
        
        count = 0
        total_count = 0
        action = np.array([0.,0.])
        dd_count = 0

        print("[control] Tracking path...")
        Sreal = []
        Areal = []
        while 1:
            state = self.obs_srv().state
            Sreal.append(state)
            if i_path == S.shape[0]-1:
                # msg.data = S[-1,:]
                self.goal = S[-1,:]
                self.ueq = A[-1]
            elif self.weightedL2(state[:], S[i_path,:]) < self.tol or (self.weightedL2(state[:], S[i_path+1,:]) < self.weightedL2(state[:], S[i_path,:]) and self.weightedL2(state[:], S[i_path+1,:]) < self.tol*3):
                i_path += 1
                # msg.data = S[i_path,:]
                self.goal = S[i_path,:]
                self.ueq = A[i_path]
                count = 0
                dd_count = 0
                self.cumError = np.array([0.,0.,0.,0.])
            # self.pub_current_goal.publish(msg)

            if self.openOrClosed:
                action = self.get_action(state, self.ueq)
            else:
                action = self.ueq

            Areal.append(action)
            self.action_srv(np.array([action])) # Command the action here

            if count > 50 or not self.valid_srv().valid_state:
                print("[control] Fail.")
                success = False
                break

            if self.wrapEuclideanNorm(state[:], S[-1,:]) < self.goal_tol:
                print("[control] Reached GOAL!!!")
                success = True
                print "State: ", state
                print "goal: ", S[-1,:]
                break

            count += 1
            total_count += 1
            # self.rate.sleep()

        Sreal = np.array(Sreal).reshape(-1)
        Areal = np.array(Areal).reshape(-1)
        return Sreal, Areal, success

    def get_action(self, state, ueq = 0.0):

        e = self.wrapEuclidean(state, self.goal)
        e = [0.0 if np.abs(ee) < 1e-4 else ee for ee in e]

        self.cumError += e
        Ki = np.array([100.,100.,10.,10.]) # case 1, 2
        # Ki = np.array([100.,100.,10.,10.])*100.

        if self.controller == 'pd':
            K = np.array([1000., 1000., 110.0, 120.0])
            action = -np.dot( K, e) - np.dot(Ki, self.cumError)
            return action.item()
        
        if self.controller == 'lqr':
            """Solve the continuous time lqr controller.
            dx/dt = A x + B u
            cost = integral x.T*Q*x + u.T*R*u
            """
            # H = 1000. # case 3, Ki = K
            H = 100. # case 1, 2, Ki
            Q = np.diag([100., 100.0, 10.0, 10.0])*H
            R = 1.*np.diag([1.])
            invR = 1./R # scipy.linalg.inv(R)
            A, B = self.linear_acrobot_system(state, ueq)
            try:
                K, _, _ = lqr(A, np.matrix(B).reshape(-1,1), Q, R)
                action = -np.dot(K, e)[0] + ueq - np.dot(Ki, self.cumError) # Added integrator action, Murray: https://www.cds.caltech.edu/~murray/courses/cds110/wi06/lqr.pdf
            except:
                rospy.logerr('[general_control] Error in action computation!')
                if np.all(np.abs(state)<1e-8):
                    action = 1e-1
                else:
                    action = 0.0

            return action

    def wrapEuclideanNorm(self, x, y):
        v = np.pi

        d = 0
        s = 0
        for i in range(len(x)):
            d = np.abs(x[i]-y[i])
            if i < 2:
                d = 2*v - d if d > v else d
            s += d**2

        return np.sqrt( s )

    def CallbackBestAction(self, msg):
        self.action = msg.data

    def linear_acrobot_system(self, x, u):
        lc = .5
        lc2 = .25
        l2 = 1.0
        m = 1.0
        I1 = 0.2
        I2 = 1.0
        l = 1.0
        g = 9.8

        theta1 = x[0] - np.pi/2
        theta2 = x[1] 
        theta1dot = x[2]
        theta2dot = x[3]

        # u = 0.0
        # u = self.planned_action
        # u = g*lc*m*cos(theta1 + theta2)

        A = np.array([[ 0, 0, 1, 0],[ 0, 0, 0, 1],
            [((g*sin(theta1)*(l*m + lc*m) + g*lc*m*sin(theta1 + theta2))*(I2 + lc2*m) - g*lc*m*sin(theta1 + theta2)*(I2 + m*(lc2 + l*lc*cos(theta2))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2),                                                                                                                                                                                                                                                                  - ((- l*lc*m*cos(theta2)*theta1dot**2 + g*lc*m*sin(theta1 + theta2))*(I2 + m*(lc2 + l*lc*cos(theta2))) - (I2 + lc2*m)*(l*lc*m*cos(theta2)*theta2dot**2 + 2*l*lc*m*theta1dot*cos(theta2)*theta2dot + g*lc*m*sin(theta1 + theta2)) + l*lc*m*sin(theta2)*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2)))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2) - ((2*l*lc*m*sin(theta2)*(I2 + lc2*m) - 2*l*lc*m*sin(theta2)*(I2 + m*(lc2 + l*lc*cos(theta2))))*((I2 + lc2*m)*(- l*lc*m*sin(theta2)*theta2dot**2 - 2*l*lc*m*theta1dot*sin(theta2)*theta2dot + theta1dot/10 + g*cos(theta1)*(l*m + lc*m) + g*lc*m*cos(theta1 + theta2)) - (I2 + m*(lc2 + l*lc*cos(theta2)))*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2)**2,                                           ((2*l*lc*m*theta2dot*sin(theta2) - 1/10)*(I2 + lc2*m) + 2*l*lc*m*theta1dot*sin(theta2)*(I2 + m*(lc2 + l*lc*cos(theta2))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2),                                                   (I2/10 + (m*(lc2 + l*lc*cos(theta2)))/10 + 2*l*lc*m*sin(theta2)*(theta1dot + theta2dot)*(I2 + lc2*m))/(I1*I2 + lc2**2*m**2 + l2*lc2*m**2 + I2*l2*m + I1*lc2*m + I2*lc2*m - l**2*lc**2*m**2*cos(theta2)**2)],
            [ -((g*sin(theta1)*(l*m + lc*m) + g*lc*m*sin(theta1 + theta2))*(I2 + m*(lc2 + l*lc*cos(theta2))) - g*lc*m*sin(theta1 + theta2)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2), - ((I2 + m*(lc2 + l*lc*cos(theta2)))*(l*lc*m*cos(theta2)*theta2dot**2 + 2*l*lc*m*theta1dot*cos(theta2)*theta2dot + g*lc*m*sin(theta1 + theta2)) - (- l*lc*m*cos(theta2)*theta1dot**2 + g*lc*m*sin(theta1 + theta2))*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - 2*l*lc*m*sin(theta2)*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2)) + l*lc*m*sin(theta2)*(- l*lc*m*sin(theta2)*theta2dot**2 - 2*l*lc*m*theta1dot*sin(theta2)*theta2dot + theta1dot/10 + g*cos(theta1)*(l*m + lc*m) + g*lc*m*cos(theta1 + theta2)))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2) - (((I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m)*(l*lc*m*sin(theta2)*theta1dot**2 + theta2dot/10 - u + g*lc*m*cos(theta1 + theta2)) - (I2 + m*(lc2 + l*lc*cos(theta2)))*(- l*lc*m*sin(theta2)*theta2dot**2 - 2*l*lc*m*theta1dot*sin(theta2)*theta2dot + theta1dot/10 + g*cos(theta1)*(l*m + lc*m) + g*lc*m*cos(theta1 + theta2)))*(2*l*lc*m*sin(theta2)*(I2 + lc2*m) - 2*l*lc*m*sin(theta2)*(I2 + m*(lc2 + l*lc*cos(theta2)))))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2)**2, -((I2 + m*(lc2 + l*lc*cos(theta2)))*(2*l*lc*m*theta2dot*sin(theta2) - 1/10) + 2*l*lc*m*theta1dot*sin(theta2)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2), -(I1/10 + I2/10 + (m*(l2 + lc2 + 2*l*lc*cos(theta2)))/10 + (lc2*m)/10 + 2*l*lc*m*sin(theta2)*(theta1dot + theta2dot)*(I2 + lc2*m + l*lc*m*cos(theta2)))/(I1*I2 + lc2**2*m**2 + l2*lc2*m**2 + I2*l2*m + I1*lc2*m + I2*lc2*m - l**2*lc**2*m**2*cos(theta2)**2)]])
 
        B = np.array([0, 0, -(I2 + m*(lc2 + l*lc*cos(theta2)))/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2), (I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m)/((I2 + lc2*m)*(I1 + I2 + m*(l2 + lc2 + 2*l*lc*cos(theta2)) + lc2*m) - (I2 + m*(lc2 + l*lc*cos(theta2)))**2)])

        return A, B

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


if __name__ == '__main__':
    try:
        general_control()
    except rospy.ROSInterruptException:
        pass