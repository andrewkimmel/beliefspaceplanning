#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import rolloutReq, plotReq, observation, IsDropped, TargetAngles
import numpy as np
import matplotlib.pyplot as plt


class rollout():

    states = []

    def __init__(self):
        rospy.init_node('rollout_node', anonymous=True)

        rospy.Service('/rollout/rollout', rolloutReq, self.CallbackRollout)
        rospy.Service('/rollout/plot', plotReq, self.Plot)

        self.obs_srv = rospy.ServiceProxy('/hand_control/observation', observation)
        self.drop_srv = rospy.ServiceProxy('/hand_control/IsObjDropped', IsDropped)
        self.move_srv = rospy.ServiceProxy('/hand_control/MoveGripper', TargetAngles)
        self.reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)

        self.state_dim = 6
        self.action_dim = 2
        self.stepSize = 10

        self.rate = rospy.Rate(15) # 15hz
        while not rospy.is_shutdown():
            rospy.spin()

    def run_rollout(self, A):
        print("Rolling-out...")

        # Reset gripper
        self.reset_srv()

        # Start episode
        S = []
        for i in range(A.shape[0]):
            # Get observation and choose action
            state = np.array(self.obs_srv().state)
            action = A[i,:]

            S.append(state)
            
            suc = True
            for _ in range(self.stepSize+1):
                suct = self.move_srv(action).success
                rospy.sleep(0.2) # For sim_data_discrete v5
                # rospy.sleep(0.05) # For all other
                self.rate.sleep()
                if not suct:
                    suc = False

            # Get observation
            next_state = np.array(self.obs_srv().state)

            if suc:
                fail = self.drop_srv().dropped # Check if dropped - end of episode
            else:
                # End episode if overload or angle limits reached
                rospy.logerr('[rollout] Failed to move gripper. Episode declared failed.')
                fail = True

            state = next_state

            if not suc or fail:
                print("[rollout] Fail")
                S.append(state)
                break

        return np.array(S)

    def CallbackRollout(self, req):
        
        actions = np.array(req.actions).reshape(-1, self.action_dim)

        self.states = self.run_rollout(actions)

        return {'states': self.states.reshape((-1,))}

    def Plot(self, req):
        planned = np.array(req.states).reshape(-1, self.state_dim)

        plt.plot(self.states[:,0], self.states[:,1],'b', label='Rolled-out path')
        plt.plot(planned[:,0], planned[:,1],'r', label='Planned path')
        plt.legend()
        plt.show()

        return EmptyResponse()


if __name__ == '__main__':
    try:
        rollout()
    except rospy.ROSInterruptException:
        pass