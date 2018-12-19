#!/usr/bin/env python

import rospy
import numpy as np
from pilco import PILCO
from controllers import RbfController, LinearController
from rewards import ExponentialReward, ExponentialRewardAxis, ExponentialRewardCustom
import matplotlib.pyplot as plt

from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from toy_simulator.srv import TargetAngles, IsDropped, observation, transition
import pickle

# np.random.seed(0)

state_dim = 2
control_dim = 2

# There is a tensorflow error if this are in the class
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
# controller = LinearController(state_dim=state_dim, control_dim=control_dim)


class pilco_node():
    # goal = np.array([0.25,0.4]) # Chase
    goal = np.array([0.95,0.95]) # Square
    max_steps = 120
    horizon = 40

    def __init__(self):
        rospy.init_node('pilco_node', anonymous=True)
        self.rate = rospy.Rate(15)

        self.reset_srv = rospy.ServiceProxy('/toy/ResetGripper', Empty)
        self.move_srv = rospy.ServiceProxy('/toy/MoveGripper', TargetAngles)
        self.obs_srv = rospy.ServiceProxy('/toy/observation', observation)
        clear_srv = rospy.ServiceProxy('/plot/clear', Empty)
        self.plot_srv = rospy.ServiceProxy('/plot/plot', Empty)
        self.clear_srv = rospy.ServiceProxy('/plot/clear', Empty)

        pub_goal = rospy.Publisher('/toy/goal', Float32MultiArray, queue_size=10)
        msg = Float32MultiArray()

        msg.data = self.goal
        # Initial random rollouts to generate a dataset
        X,Y,_ = self.rollout(policy=self.random_policy, steps=self.horizon)
        for i in range(1,10):
            X_, Y_, suc = self.rollout(policy=self.random_policy, steps=self.horizon)
            if not suc:
                continue
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))

        data_size = X.shape[0]
        state_dim = Y.shape[1]
        control_dim = X.shape[1] - state_dim
        # controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
        # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

        # pilco = PILCO(X, Y, controller=controller, horizon=40)
        # Example of user provided reward function, setting a custom target state
        W = np.eye(2)
        R = ExponentialReward(state_dim=state_dim, t=self.goal, W=W)
        # R = ExponentialRewardCustom(state_dim=state_dim, t=self.goal)
        # R = ExponentialRewardAxis(state_dim=state_dim, t=self.goal)
        self.pilco = PILCO(X, Y, controller=controller, horizon=self.horizon, reward=R, num_induced_points=10)#int(data_size/10))

        # Example of fixing a parameter, optional, for a linear controller only
        #pilco.controller.b = np.array([[0.0]])
        #pilco.controller.b.trainable = False

        Iter = 0
        success_count = 0
        fail_count = 0
        last_convg = np.array([0.,0.])
        thr = 0.05
        add_action_noise = False
        while not rospy.is_shutdown():
            pub_goal.publish(msg)

            Iter += 1
            self.pilco.optimize()
            # import pdb; pdb.set_trace()
            clear_srv()
            X_new, Y_new, suc = self.rollout(policy=self.pilco_policy, steps=self.max_steps, add_action_noise=add_action_noise)
            if not suc:
                self.rate.sleep()
                continue

            add_action_noise = False
            # Update dataset
            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            self.pilco.mgpr.set_XY(X, Y)
            cur_pos = X_new[-1,:2]+Y_new[-1,:]
            print('[pilco_node] Iteration ' + str(Iter) + ', reached:' + str(cur_pos) + ', distance: ' + str( np.linalg.norm((cur_pos)-self.goal) ))
            if np.linalg.norm(self.goal-cur_pos) < thr:
                print('[pilco_node] Goal reached after %d iterations, %d/3 trial.!'%(Iter, success_count+1))
                print('')
                success_count += 1
                if success_count >= 3:
                    break
            else:
                success_count = 0
                fail_count += 1
                print('[pilco_node] Missed goal, %d/3 trial.!'%(fail_count))
                # if fail_count >=3 and np.linalg.norm(last_convg-cur_pos) < 2*thr:
                #     add_action_noise = True
            last_convg = cur_pos


            # if Iter >= 30:
            #     break
            
            self.rate.sleep()

        X, Y = self.rollout(policy=self.pilco_policy, steps=self.max_steps)
        plt.figure()
        plt.plot(X[:,0], X[:,1],'-k')
        plt.plot(X[0,0], X[0,1],'or')
        plt.plot(self.goal[0], self.goal[1],'og')
        xend = X[-1,:2] + Y[-1,:]
        plt.plot(xend[0], xend[1],'ob')
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
            
    def rollout(self, policy, steps, add_action_noise=False):
        X = []; Y = []
        print('[pilco_node] Rollout...')        

        # Reset system
        self.reset_srv()
        self.clear_srv()

        # Get observation
        x = np.array(self.obs_srv().state)

        for i in range(steps):
            # env.render()
            
            u = policy(x)
            if add_action_noise:
                print('Adding action noise to u = ' + str(u))
                u += np.random.normal(0., 0.1, 2)
                print('new u = ' + str(u))

            # Act
            suc = self.move_srv(u).success

            if suc: # Get observation
                x_new = np.array(self.obs_srv().state)
            else: # End episode if overload or angle limits reached
                rospy.logerr('[pilco_node] Failed to move. Episode declared failed.')
                break
            
            X.append(np.hstack((x, u)))
            Y.append(x_new - x)
            x = x_new

            self.rate.sleep()

        print('[pilco_node] Episode ended')
        self.plot_srv()

        suc = True
        if len(X)==0:
            suc = False 
            X = np.array([0.,0.,0.,0.])
            Y = np.array([0.,0.])

        return np.stack(X), np.stack(Y), suc

    def random_policy(self, x):
        return np.random.uniform(-1.,1.,2)

    def pilco_policy(self, x):
        return self.pilco.compute_action(x[None, :])[0, :]




if __name__ == '__main__':
    try:
        pilco_node()
    except rospy.ROSInterruptException:
        pass

