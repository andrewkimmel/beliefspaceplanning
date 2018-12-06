#!/usr/bin/env python

import rospy
import numpy as np
from pilco import PILCO
from controllers import RbfController, LinearController
from rewards import ExponentialReward

from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from ChaseProbEnv import ChaseEnv, ProbEnv
from toy_simulator.srv import TargetAngles, IsDropped, observation, transition

np.random.seed(0)
goal = np.array([0.,-0.7])

class pilco_node():

    def __init__(self):
        self.reset_srv = rospy.ServiceProxy('/toy/ResetGripper', Empty)
        self.move_srv = rospy.ServiceProxy('/toy/MoveGripper', TargetAngles)
        rospy.ServiceProxy('/toy/IsObjDropped', IsDropped)
        self.obsv_srv = rospy.ServiceProxy('/toy/observation', observation)
        rospy.ServiceProxy('/toy/transition', transition)

        rospy.init_node('pilco_node', anonymous=True)
        self.rate = rospy.Rate(15)

        data_size = X.shape[0]

        state_dim = Y.shape[1]
        control_dim = X.shape[1] - state_dim
        controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
        #controller = LinearController(state_dim=state_dim, control_dim=control_dim)

        # pilco = PILCO(X, Y, controller=controller, horizon=40)
        # Example of user provided reward function, setting a custom target state
        R = ExponentialReward(state_dim=state_dim, t=goal)
        pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R, num_induced_points=int(data_size/10))

        # Example of fixing a parameter, optional, for a linear controller only
        #pilco.controller.b = np.array([[0.0]])
        #pilco.controller.b.trainable = False

        # Initial random rollouts to generate a dataset
        X,Y = rollout(policy=self.random_policy, steps=40)
        for i in range(1,10):
        X_, Y_ = rollout(policy=self.random_policy, steps=40)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))

        iter = 0
        while not rospy.is_shutdown():
            iter += 1
            pilco.optimize()
            # import pdb; pdb.set_trace()
            X_new, Y_new = self.rollout(policy=self.pilco_policy, steps=100)
            # Update dataset
            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_XY(X, Y)
            print('[pilco_node] Iteration ' + str(iter) + ', Reached:' + str(X[-1,:2]))
            if np.linalg.norm(goal-X[-1,:2]) < 0.03:
                print('[pilco_node] Goal reached after %d iterations!'%iter)
                break
            
            self.rate.sleep()

    def rollout(self, policy, steps):
        X = []; Y = []

        # Reset system
        reset_srv()

        # Get observation
        x = np.array(obs_srv().state)

        for step in range(steps):
            # env.render()
            
            u = policy(x)
            
            # Act
            suc = move_srv(action[:2])

            if suc:
                # Get observation
                x_new = np.array(obs_srv().state)
            else:
                # End episode if overload or angle limits reached
                rospy.logerr('[pilco_node] Failed to move. Episode declared failed.')
                break
            
            X.append(np.hstack((x, u)))
            Y.append(x_new - x)
            x = x_new
            
            self.rate.sleep()

        return np.stack(X), np.stack(Y)

    def random_policy(x):
        return env.action_sample()

    def pilco_policy(x):
        return pilco.compute_action(x[None, :])[0, :]





