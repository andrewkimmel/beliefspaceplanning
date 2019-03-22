#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse, SetBool
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
    pi_cross = False

    def __init__(self):
        rospy.init_node('rollout_node', anonymous=True)

        rospy.Service('/rollout/rollout', rolloutReq, self.CallbackRollout)
        rospy.Service('/rollout/rollout_from_file', rolloutReqFile, self.CallbackRolloutFile)
        rospy.Service('/rollout/plot', plotReq, self.Plot)
        self.action_pub = rospy.Publisher('/rollout/gripper_action', Float32MultiArray, queue_size = 10)
        rospy.Subscriber('/acrobot/pi_cross', Bool, self.callbackCross)

        self.obs_srv = rospy.ServiceProxy('/acrobot_control/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/acrobot_control/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/acrobot_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/acrobot_control/ResetGripper', Empty)

        self.trigger_srv = rospy.ServiceProxy('/rollout_recorder/trigger', SetBool)
        self.trigger_actor_srv = rospy.ServiceProxy('/rollout_actor/trigger', SetBool)
        self.gets_srv = rospy.ServiceProxy('/rollout_recorder/get_states', gets)

        self.state_dim = var.state_dim_
        self.action_dim = var.state_action_dim_-var.state_dim_
        self.stepSize = var.stepSize_ # !!!!!!!!!!!!!!!!!!!!!!!!!!

        print("[rollout] Ready to rollout...")

        self.rate = rospy.Rate(100) 
        # while not rospy.is_shutdown():
        rospy.spin()

    def run_rollout(self, A):
        self.rollout_transition = []

        # Reset
        self.reset_srv()

        print("[rollout] Rolling-out...")

        msg = Float32MultiArray()

        # Start episode
        success = True
        self.trigger_srv(True)
        self.trigger_actor_srv(True)
        stepSize = var.stepSize_
        n = 0
        i = 0
        while 1:
            if n == 0:
                action = A[i]
                i += 1
                n = stepSize
            
            msg.data = action
            self.action_pub.publish(msg)
            n -= 1
            print action
            
            if self.drop_srv().dropped or self.pi_cross:
                print("[rollout] Fail.")
                success = False
                break
            
            if i == A.shape[0] and n == 0:
                print("[rollout] Complete.")
                success = True
                break

            self.rate.sleep()

        self.trigger_actor_srv(False)
        self.trigger_srv(False)
        print("[rollout] Rollout done.")

        # self.states = np.array(S).reshape((-1,))
        # return success
        
        SA = self.gets_srv()
        self.states = SA.states
        self.actions = []
        
        return success

    def callbackCross(self, msg):
        self.pi_cross = np.array(msg.data)

    def CallbackRollout(self, req):
        
        actions_nom = np.array(req.actions).reshape(-1, self.action_dim)
        success = self.run_rollout(actions_nom)

        return {'states': self.states, 'actions_res': self.actions, 'success' : success}

    def CallbackRolloutFile(self, req):

        file_name = req.file

        actions = np.loadtxt(file_name, delimiter=',', dtype=float)[:,:2]
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