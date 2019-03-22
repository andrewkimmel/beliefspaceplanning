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
sys.path.insert(0, '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class general_control():

    drop = True
    obj_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    action = np.array([0., 0.])
    gripper_closed = 'open'
    tol = 1.5
    goal_tol = 10.0
    horizon = 1

    def __init__(self):
        rospy.init_node('control', anonymous=True)

        self.gp = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)

        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)
        rospy.Subscriber('/hand_control/gripper_status', String, self.callbackGripperStatus)

        rospy.Service('/control', pathTrackReq, self.CallbackTrack)
        rospy.Subscriber('/gp_controller/action', Float32MultiArray, self.CallbackBestAction)
        # rospy.Subscriber('/vs_controller/action', Float32MultiArray, self.CallbackBestAction)
        self.pub_current_goal = rospy.Publisher('/control/goal', Float32MultiArray, queue_size=10)
        self.pub_horizon = rospy.Publisher('/control/horizon', Float32MultiArray, queue_size=10)
        self.pub_exclude = rospy.Publisher('/control/exclude', Float32MultiArray, queue_size=10)

        self.obs_srv = rospy.ServiceProxy('/hand_control/observation', observation)
        self.move_srv = rospy.ServiceProxy('/hand_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)

        self.trigger_srv = rospy.ServiceProxy('/rollout_recorder/trigger', Empty)
        self.gets_srv = rospy.ServiceProxy('/rollout_recorder/get_states', gets)

        self.plot_clear_srv = rospy.ServiceProxy('/plot/clear', Empty)
        self.plot_ref_srv = rospy.ServiceProxy('/plot/ref', pathTrackReq)

        self.state_dim = var.state_dim_
        self.action_dim = var.state_action_dim_-var.state_dim_
        self.stepSize = 5# var.stepSize_

        print("[control] Ready to track path...")
        self.rate = rospy.Rate(2) 
        rospy.spin()

    def CallbackTrack(self, req):

        path = np.array(req.desired_path).reshape(-1, self.state_dim)
        
        real_path, success = self.run_tracking(path)

        return {'real_path': real_path, 'success' : success}

    def weightedL2(self, ds, W = np.diag(np.array([1.,1.,0.2, 0.2]))):
        # return np.sqrt( np.dot(ds.T, np.dot(W, ds)) )
        return np.linalg.norm(ds[:2])

    def run_tracking(self, S):
        
        # Reset gripper
        while 1:
            self.reset_srv()
            while not self.gripper_closed:
                self.rate.sleep()

            # Verify starting state
            if np.abs(self.obj_pos[0]-3.30313851e-02) < 0.01831497*2. and np.abs(self.obj_pos[1]-1.18306790e+02) < 0.10822673*2.:
                break

        i_path = 1
        msg = Float32MultiArray()
        msg.data = S[i_path,:]
        msge = Float32MultiArray()
        self.pub_current_goal.publish(msg)
        self.rate.sleep()
        
        self.plot_clear_srv()
        self.plot_ref_srv(S.reshape((-1,)))
        self.trigger_srv()
        n = -1
        count = 0
        total_count = 0
        d_prev = 1000
        action = np.array([0.,0.])
        dd_count = 0
        
        print("[control] Tracking path...")
        while 1:
            change = False
            state = np.concatenate((self.obj_pos, self.gripper_load), axis=0)
            if i_path == S.shape[0]-1:
                msg.data = S[-1,:]
            elif self.weightedL2(state[:]-S[i_path,:]) < self.tol or (self.weightedL2(state[:]-S[i_path+1,:]) < self.weightedL2(state[:]-S[i_path,:]) and self.weightedL2(state[:]-S[i_path+1,:]) < self.tol*2):
                i_path += 1
                msg.data = S[i_path,:]
                count = 0
                change = True
                self.tol = 1.5
                dd_count = 0
            elif count > 100:# and i_path < S.shape[0]-1:
                self.tol = 3
            self.pub_current_goal.publish(msg)

            dd = self.weightedL2(state[:]-S[i_path,:]) - d_prev
            dd_count = dd_count + 1 if dd > 0 else 0
            msge.data = action if dd_count > 3 else np.array([0.,0.])
            self.pub_exclude.publish(msge)

            if n == 0 and not change and dd < 0:
                n = 1
                print "Extended..."
            if n <= 0 or dd_count > 5:
                action = self.action
                n = self.stepSize
                dd_count = 0

            print total_count, count, i_path, action, self.weightedL2(state[:]-S[i_path,:]), self.weightedL2(state[:]-S[-1,:]), self.weightedL2(state[:]-S[i_path,:]) - d_prev
            
            d_prev =  self.weightedL2(state[:]-S[i_path,:])

            suc = self.move_srv(action).success
            n -= 1

            if not suc or self.drop or count > 400:
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

        Sreal = self.gets_srv().states

        return Sreal, success

    def callbackDrop(self, msg):
        self.drop = msg.data

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def CallbackBestAction(self, msg):
        self.action = np.array(msg.data)

    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"


if __name__ == '__main__':
    try:
        general_control()
    except rospy.ROSInterruptException:
        pass