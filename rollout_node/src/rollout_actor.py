#!/usr/bin/env python

import rospy
import numpy as np
import time
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles

class rolloutAct():
    running = False
    action = np.array([0.,0.])
    
    def __init__(self):
                
        rospy.init_node('rollout_actor', anonymous=True)

        rospy.Subscriber('/rollout/gripper_action', Float32MultiArray, self.callbackAction)
        rospy.Service('/rollout_actor/trigger', Empty, self.callbackTrigger)
        move_srv = rospy.ServiceProxy('/acrobot_control/MoveGripper', TargetAngles)
        drop_srv = rospy.ServiceProxy('/acrobot_control/IsObjDropped', IsDropped)

        rate = rospy.Rate(2)
        while not rospy.is_shutdown():

            if self.running:
                suc = move_srv(self.action).success
                
                if not suc or drop_srv().dropped:
                    print('[rollout_actor] Episode ended.')
                    self.running = False

            rate.sleep()

    def callbackAction(self, msg):
        self.action = np.array(msg.data)

    def callbackTrigger(self, msg):
        self.running = True

        return EmptyResponse()

       
if __name__ == '__main__':
    
    try:
        rolloutAct()
    except rospy.ROSInterruptException:
        pass