#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import observation, IsDropped, TargetAngles
from gpup_gp_node.srv import one_transition
from control.srv import pathTrackReq
from rollout_node.srv import gets

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class general_control():

    drop = True
    state = np.array([0., 0., 0., 0.])
    action = np.array([0.])
    tol = 0.2
    goal_tol = 0.2
    horizon = 1

    def __init__(self):
        rospy.init_node('control', anonymous=True)

        self.gp = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

        rospy.Subscriber('/acrobot/state', Float32MultiArray, self.callbackState)

        rospy.Service('/control', pathTrackReq, self.CallbackTrack)
        rospy.Subscriber('/controller/action', Float32MultiArray, self.CallbackBestAction)
        self.pub_current_goal = rospy.Publisher('/control/goal', Float32MultiArray, queue_size=10)

        self.plot_clear_srv = rospy.ServiceProxy('/plot/clear', Empty)
        self.plot_ref_srv = rospy.ServiceProxy('/plot/ref', pathTrackReq)

        self.state_dim = var.state_dim_
        self.action_dim = var.state_action_dim_-var.state_dim_
        self.stepSize = var.stepSize_

        print("[control] Ready to track path...")
        self.rate = rospy.Rate(2) 
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
        #### ???? ####

        i_path = 1
        msg = Float32MultiArray()
        msg.data = S[i_path,:]
        msge = Float32MultiArray()
        for i in range(5):
            self.pub_current_goal.publish(msg)
            self.rate.sleep()

        rospy.sleep(1.0)
        
        self.plot_clear_srv()
        self.plot_ref_srv(S.reshape((-1,)))
        
        # Trigger recording here

        count = 0
        total_count = 0
        action = np.array([0.,0.])
        dd_count = 0
        
        print("[control] Tracking path...")
        while 1:
            state = self.state
            if i_path == S.shape[0]-1:
                msg.data = S[-1,:]
            elif self.weightedL2(state[:]-S[i_path,:]) < self.tol or (self.weightedL2(state[:]-S[i_path+1,:]) < self.weightedL2(state[:]-S[i_path,:]) and self.weightedL2(state[:]-S[i_path+1,:]) < self.tol*3):
                i_path += 1
                msg.data = S[i_path,:]
                count = 0
                dd_count = 0
            self.pub_current_goal.publish(msg)

            action = self.action
            print total_count, count, i_path, action, self.weightedL2(state[:]-S[i_path,:]), self.weightedL2(state[:]-S[-1,:])
            
            suc = self.move_srv(action).success # Command the action here

            if not suc or self.drop or count > 1000:
                print("[control] Fail.")
                success = False
                break

            if np.linalg.norm(state[:2]-S[-1,:2]) < self.goal_tol:
                print("[control] Reached GOAL!!!")
                success = True
                break

            count += 1
            total_count += 1
            self.rate.sleep()

        # return Sreal, Areal, success

    def callbackState(self, msg):
        self.state = np.array(msg.data)

    def CallbackBestAction(self, msg):
        self.action = np.array(msg.data)



if __name__ == '__main__':
    try:
        general_control()
    except rospy.ROSInterruptException:
        pass