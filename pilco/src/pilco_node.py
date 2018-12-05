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
        rospy.Service('/toy/ResetGripper', Empty)
        rospy.Service('/toy/MoveGripper', TargetAngles)
        rospy.Service('/toy/IsObjDropped', IsDropped)
        rospy.Service('/toy/observation', observation)
        rospy.Service('/toy/transition', transition)

        rospy.init_node('pilco_node', anonymous=True)







def rollout(policy, timesteps):
    X = []; Y = []
    env.reset()
    x, _, _ = env.step([0.,0.])
    for timestep in range(timesteps):
        env.render()
        u = policy(x)
        x_new, _, done = env.step(u)
        if done: break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)

def random_policy(x):
    return env.action_sample()

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]

# Initial random rollouts to generate a dataset
X,Y = rollout(policy=random_policy, timesteps=40)
for i in range(1,10):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

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

iter = 0
while 1:
    iter += 1
    pilco.optimize()
    # import pdb; pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=100)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
    print('Iteration ' + str(iter) + ', Reached:' + str(X[-1,:2]))
    if np.linalg.norm(goal-X[-1,:2]) < 0.03:
        print('Goal reached after %d iterations!'%iter)
        break

X, Y = rollout(policy=pilco_policy, timesteps=100)
env.plot()

