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
    obj_pos = np.array([0., 0.])
    gripper_load = np.array([0., 0.])
    actionGP = np.array([0., 0.])
    actionNN = np.array([0., 0.])
    actionVS = np.array([0., 0.])
    gripper_closed = 'open'
    tol = 0.3
    goal_tol = 0.3
    horizon = 1

    def __init__(self):
        rospy.init_node('control', anonymous=True)

        self.gp = rospy.ServiceProxy('/gp/transitionOneParticle', one_transition)
        self.nn = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)

        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        rospy.Subscriber('/gripper/load', Float32MultiArray, self.callbackGripperLoad)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)
        rospy.Subscriber('/hand_control/gripper_status', String, self.callbackGripperStatus)

        rospy.Service('/control', pathTrackReq, self.CallbackTrack)
        rospy.Subscriber('/gp_controller/action', Float32MultiArray, self.CallbackBestActionGP)
        rospy.Subscriber('/nn_controller/action', Float32MultiArray, self.CallbackBestActionNN)
        rospy.Subscriber('/vs_controller/action', Float32MultiArray, self.CallbackBestActionVS)
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
        self.stepSize = var.stepSize_

        print("[control] Ready to track path...")
        self.rate = rospy.Rate(2) 
        rospy.spin()

    def CallbackTrack(self, req):

        path = np.array(req.desired_path).reshape(-1, 2)
        
        real_path, actions, success = self.run_tracking(path)

        return {'real_path': real_path, 'actions': actions, 'success' : success}

    def weightedL2(self, ds, W = np.diag(np.array([1.,1.,0.2, 0.2]))):
        # return np.sqrt( np.dot(ds.T, np.dot(W, ds)) )
        return np.linalg.norm(ds[:2])

    def run_tracking(self, S):
        
        fail = True
        N = 0
        a = np.array([-1,1])
        while fail:
            # Reset gripper
            self.reset_srv()
            break

            # Shoot
            print('[general control] Shooting...')
            for i in range(N):
                suc = self.move_srv(a).success
                if not suc or self.drop:
                    break
                self.rate.sleep()

                if i == N-1:
                    fail = False
            

        # Move relative to current position
        S += self.obj_pos

        i_path = 0
        msg = Float32MultiArray()
        msg.data = S[i_path,:]
        msge = Float32MultiArray()
        for i in range(5):
            self.pub_current_goal.publish(msg)
            self.rate.sleep()

        rospy.sleep(1.0)
        
        self.plot_clear_srv()
        self.plot_ref_srv(S.reshape((-1,)))
        self.trigger_srv()
        n = -1
        count = 0
        total_count = 0
        d_prev = 1000
        action = np.array([0.,0.])
        excluded_action = np.array([0.,0.])
        dd_count = 0
        Controller = 'NN'
        sk = 2
        
        print("[control] Tracking path...")
        start_time = rospy.get_time()
        while 1:
            change = False
            state = np.concatenate((self.obj_pos, self.gripper_load), axis=0)
            if i_path == S.shape[0]-1:
                msg.data = S[-1,:]
            elif self.weightedL2(state[:sk]-S[i_path,:sk]) < self.tol or (self.weightedL2(state[:sk]-S[i_path+1,:sk]) < self.weightedL2(state[:sk]-S[i_path,:sk])):# and self.weightedL2(state[:sk]-S[i_path+1,:sk]) < self.tol*2.5):
                i_path += 1
                msg.data = S[i_path,:]
                count = 0
                change = True
                dd_count = 0
                print "Distance to next waypoint: " + str(self.weightedL2(state[:sk]-S[i_path,:sk])) + ", moving to the next one # " + str(i_path) + "."
            #     Controller == 'NN'
            # elif count > 150:# and i_path < S.shape[0]-1:
            #     # self.tol = 2.5
            #     Controller == 'VS'
            self.pub_current_goal.publish(msg)

            dd = self.weightedL2(state[:sk]-S[i_path,:sk]) - d_prev
            dd_count = dd_count + 1 if dd > 0.  else 0
            if np.any(action != np.array([0.,0.])):
                excluded_action = action if dd_count > 10 else np.array([0.,0.])
            msge.data = excluded_action
            self.pub_exclude.publish(msge)

            if n <= 0:# or dd_count > 5:
                # if 0 and dd_count <= 7:
                if 0 and Controller == 'NN':
                    action = self.actionNN
                    if np.all(action == excluded_action):
                        action = np.array([0.,0.])
                    else:
                        dd_count = 0
                else:
                    action = self.actionVS
                n = self.stepSize
                dd_count = 0

            print "\n"
            print total_count, count, i_path, action, dd_count, excluded_action, self.weightedL2(state[:sk]-S[i_path,:sk]), self.weightedL2(state[:sk]-S[-1,:sk]), self.weightedL2(state[:sk]-S[i_path,:sk]) - d_prev
            
            d_prev =  self.weightedL2(state[:sk]-S[i_path,:sk])

            suc = self.move_srv(action).success
            n -= 1

            if not suc or self.drop or count > 1000:
                print("[control] Fail.")
                success = False
                break

            if np.linalg.norm(state[:sk]-S[-1,:sk]) < self.goal_tol and S.shape[0] - i_path < 3:
                print("[control] Reached GOAL!!!")
                T = rospy.get_time() - start_time
                print('[control] Runtime: %f'%T)
                success = True
                break

            # if total_count > 100:
            #     success = True
            #     break

            count += 1
            total_count += 1
            self.rate.sleep()

        Sreal = self.gets_srv().states
        Areal = self.gets_srv().actions

        return Sreal, Areal, success

    def callbackDrop(self, msg):
        self.drop = msg.data

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def callbackGripperLoad(self, msg):
        self.gripper_load = np.array(msg.data)

    def CallbackBestActionGP(self, msg):
        self.actionGP = np.array(msg.data)
    
    def CallbackBestActionNN(self, msg):
        self.actionNN = np.array(msg.data)

    def CallbackBestActionVS(self, msg):
        self.actionVS = np.array(msg.data)

    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"


if __name__ == '__main__':
    try:
        general_control()
    except rospy.ROSInterruptException:
        pass