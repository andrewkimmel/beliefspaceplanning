#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
from rollout_node.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import gets

import sys
sys.path.insert(0, '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
import var

class rollout():

    states = []
    actions = []
    plot_num = 0
    drop = True
    obj_pos = np.array([0., 0.])

    def __init__(self):
        rospy.init_node('rollout_node', anonymous=True)

        rospy.Service('/rollout/rollout', rolloutReq, self.CallbackRollout)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)
        rospy.Service('/rollout/plot', plotReq, self.Plot)
        rospy.Subscriber('/hand_control/cylinder_drop', Bool, self.callbackDrop)
        rospy.Subscriber('/hand_control/gripper_status', String, self.callbackGripperStatus)
        rospy.Subscriber('/hand/obj_pos', Float32MultiArray, self.callbackObj)
        self.action_pub = rospy.Publisher('/rollout/gripper_action', Float32MultiArray, queue_size = 10)

        self.obs_srv = rospy.ServiceProxy('/hand_control/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/hand_control/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/hand_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)

        self.trigger_srv = rospy.ServiceProxy('/rollout_recorder/trigger', Empty)
        self.gets_srv = rospy.ServiceProxy('/rollout_recorder/get_states', gets)

        self.state_dim = var.state_dim_
        self.action_dim = var.state_action_dim_-var.state_dim_
        self.stepSize = var.stepSize_ # !!!!!!!!!!!!!!!!!!!!!!!!!!

        print("[rollout] Ready to rollout...")

        self.rate = rospy.Rate(2) 
        # while not rospy.is_shutdown():
        rospy.spin()

    def run_rollout(self, A):
        self.rollout_transition = []

        # Reset gripper
        while 1:
            self.reset_srv()
            while not self.gripper_closed:
                self.rate.sleep()

            # Verify starting state
            # if np.abs(self.obj_pos[0]-3.30313851e-02) < 0.01831497*2. and np.abs(self.obj_pos[1]-1.18306790e+02) < 0.10822673*2.:
            break
        
        print("[rollout] Rolling-out...")

        msg = Float32MultiArray()

        # Start episode
        success = True
        state = np.array(self.obs_srv().state)
        S = []
        S.append(np.copy(state))
        self.trigger_srv()
        stepSize = var.stepSize_
        n = 0
        i = 0
        while 1:

            if n == 0:
                action = A[i,:]
                i += 1
                n = stepSize
                # print i
            
            msg.data = action
            self.action_pub.publish(msg)
            suc = self.move_srv(action).success
            n -= 1
            
            next_state = np.array(self.obs_srv().state)

            if suc:
                fail = self.drop # self.drop_srv().dropped # Check if dropped - end of episode
            else:
                # End episode if overload or angle limits reached
                rospy.logerr('[rollout] Failed to move gripper. Episode declared failed.')
                fail = True

            if n == 0 or (not suc or fail):
                # print len(S), len(self.rollout_transition)
                S.append(np.copy(next_state))
                self.rollout_transition += [(state, action, next_state, not suc or fail)]

            state = np.copy(next_state)

            if not suc or fail:
                print("[rollout] Fail.")
                success = False
                break
            
            if i == A.shape[0] and n == 0:
                print("[rollout] Complete.")
                success = True
                break

            self.rate.sleep()

        # file_pi = open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/rollout_tmp.pkl', 'wb')
        # pickle.dump(self.rollout_transition, file_pi)
        # file_pi.close()

        print("[rollout] Rollout done.")

        self.states = np.array(S).reshape((-1,))
        return success
        
        # SA = self.gets_srv()
        # self.states = SA.states
        # self.actions = SA.actions # Actions from recorder are different due to freqency difference
        
        # return success

    def callbackGripperStatus(self, msg):
        self.gripper_closed = msg.data == "closed"

    def callbackDrop(self, msg):
        self.drop = msg.data

    def callbackObj(self, msg):
        Obj_pos = np.array(msg.data)
        self.obj_pos = Obj_pos[:2] * 1000

    def CallbackRollout(self, req):
        
        actions_nom = np.array(req.actions).reshape(-1, self.action_dim)
        success = True
        success = self.run_rollout(actions_nom)

        return {'states': self.states, 'actions_res': self.actions, 'success' : success}

    def CallbackRolloutFile(self, req):

        file_name = req.file

        actions = np.loadtxt(file_name, delimiter=',', dtype=float)[:,:2]
        success = True
        success = self.run_rollout(actions)

        return {'states': self.states.reshape((-1,)), 'success' : success}

    def Plot(self, req):
        planned = np.array(req.states).reshape(-1, self.state_dim)
        plt.clf()
        plt.plot(self.states[:,0], self.states[:,1],'b', label='Rolled-out path')
        plt.plot(planned[:,0], planned[:,1],'r', label='Planned path')
        # plt.legend()
        if (req.filename):
            plt.savefig(req.filename, bbox_inches='tight')
        else:
            plt.show()

        return EmptyResponse()


if __name__ == '__main__':
    try:
        rollout()
    except rospy.ROSInterruptException:
        pass