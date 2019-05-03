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

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class general_control():

    drop = True
    state = np.array([0., 0., 0., 0.])
    action = 0.
    tol = 0.02
    goal_tol = 0.07
    horizon = 1

    def __init__(self):
        rospy.init_node('control', anonymous=True)

        rospy.Service('/control', pathTrackReq, self.CallbackTrack)
        rospy.Subscriber('/controller/action', Float32, self.CallbackBestAction)
        self.pub_current_goal = rospy.Publisher('/control/goal', Float32MultiArray, queue_size=10)

        self.plot_clear_srv = rospy.ServiceProxy('/plot/clear', Empty)
        self.plot_ref_srv = rospy.ServiceProxy('/plot/ref', pathTrackReq)

        self.obs_srv = rospy.ServiceProxy('/getObservation', simulation_observation_srv)
        self.reset_srv = rospy.ServiceProxy('/reset', simulation_reset_srv)
        self.action_srv = rospy.ServiceProxy('/sendAction', simulation_action_srv)
        self.valid_srv = rospy.ServiceProxy('/isValid', simulation_valid_srv)

        self.state_dim = var.state_dim_
        self.action_dim = var.state_action_dim_-var.state_dim_
        self.stepSize = var.stepSize_

        print("[control] Ready to track path...")
        self.rate = rospy.Rate(100) 
        rospy.spin()

    def CallbackTrack(self, req):

        path = np.array(req.desired_path).reshape(-1, self.state_dim)
        
        real_path, actions, success = self.run_tracking(path)

        return {'real_path': real_path, 'actions': actions, 'success' : success}

    def weightedL2(self, ds, W = np.diag(np.array([1.,1.,0.2, 0.2]))):
        # return np.sqrt( np.dot(ds.T, np.dot(W, ds)) )
        return np.linalg.norm(ds[:2])

    def run_tracking(self, S):
        
        # Reset Acrobot
        self.reset_srv()

        i_path = 1
        msg = Float32MultiArray()
        msg.data = S[i_path,:]
        msge = Float32MultiArray()
        for i in range(2):
            self.pub_current_goal.publish(msg)
            self.rate.sleep()

        rospy.sleep(1.0)
        
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
                msg.data = S[-1,:]
            elif self.weightedL2(state[:]-S[i_path,:]) < self.tol or (self.weightedL2(state[:]-S[i_path+1,:]) < self.weightedL2(state[:]-S[i_path,:]) and self.weightedL2(state[:]-S[i_path+1,:]) < self.tol*3):
                i_path += 1
                msg.data = S[i_path,:]
                count = 0
                dd_count = 0
            self.pub_current_goal.publish(msg)

            action = self.action
            print total_count, count, i_path, action, self.wrapEuclidean(state[:2], S[i_path,:2]), self.wrapEuclidean(state[:2], S[-1,:2])

            Areal.append(action)
            self.action_srv(np.array([action])) # Command the action here

            if count > 200:# not self.valid_srv().valid_state or count > 1000:
                print("[control] Fail.")
                success = False
                break

            if self.wrapEuclidean(state[:2], S[-1,:2]) < self.goal_tol:
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

    def wrapEuclidean(self, x, y):
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


if __name__ == '__main__':
    try:
        general_control()
    except rospy.ROSInterruptException:
        pass