#!/usr/bin/python 

import rospy
import numpy as np 
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String, Bool, Float32
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import TargetAngles, IsDropped, observation
from acrobot_controllers.srv import moveservo
import math

class acrobot_control():

    move_step_size = 0.05

    load = 0
    joint_states = np.array([0., 0.])
    joint_states_prev = np.array([0., 0.])
    joint_velocities = np.array([0., 0.])
    OBS = True
    pi_cross = False

    def __init__(self):
        rospy.init_node('acrobot_control_sim', anonymous=True)
        
        rospy.Subscriber('/acrobot/load', Float32, self.callbackLoad)
        rospy.Subscriber('/acrobot/my_joint_states', Float32MultiArray, self.callbackJoints)
        rospy.Subscriber('/acrobot/my_joint_velocities', Float32MultiArray, self.callbackJointsVel)

        rospy.Service('/acrobot_control/ResetGripper', Empty, self.ResetGripper)
        rospy.Service('/acrobot_control/MoveGripper', TargetAngles, self.MoveGripper)
        rospy.Service('/acrobot_control/IsObjDropped', IsDropped, self.CheckDropped)
        rospy.Service('/acrobot_control/observation', observation, self.GetObservation)

        self.pi_cross_pub = rospy.Publisher('/acrobot/pi_cross', Bool, queue_size=10)

        self.move_servos_srv = rospy.ServiceProxy('/MoveServos', moveservo)
        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.rate = rospy.Rate(100)
        rospy.spin()

    def callbackLoad(self, msg):
        self.load = msg.data

    def callbackJoints(self, msg):
        # self.joint_states_prev = np.copy(self.joint_states)
        self.joint_states = np.array(msg.data)

        # f = 1.5
        # if ( self.joint_states_prev[0] * self.joint_states[0] < 0 and np.abs(self.joint_states[0]) > f ) or ( self.joint_states_prev[1] * self.joint_states[1] < 0 and np.abs(self.joint_states[1]) > f ) :
        #     self.pi_cross = True

        # self.pi_cross_pub.publish(self.pi_cross)
        self.rate.sleep()

    def callbackJointsVel(self, msg):
        self.joint_velocities = np.array(msg.data)

    def ResetGripper(self, msg):
        print('[acrobot_control_sim] Resetting acrobot...')

        self.pi_cross = False

        self.move_servos_srv.call(np.array([0.0]))
        rospy.sleep(1.0)
        self.reset_srv.call()
        rospy.sleep(1.0)

        self.pi_cross = False
        
        return EmptyResponse()

    def MoveGripper(self, msg):

        desired = np.array(msg.angles)[0]
        suc = self.moveGripper(desired)

        return {'success': not self.pi_cross}
    
    def moveGripper(self, action_load):

        self.move_servos_srv.call(action_load)

        return True

    def CheckDropped(self, msg):

        # print np.abs(self.joint_velocities), any(np.abs(self.joint_velocities) >= 15.)

        if any(np.abs(self.joint_velocities) >= 8.): # np.abs(action_load) > 0.8:
            rospy.logerr('[acrobot_control_sim] Velocities to high.')
            return {'dropped': True}

        return {'dropped': False}

    def GetObservation(self, msg):
        obs = np.concatenate((self.joint_states, self.joint_velocities), axis=0)

        return {'state': obs}


if __name__ == '__main__':
    
    try:
        acrobot_control()
    except rospy.ROSInterruptException:
        pass