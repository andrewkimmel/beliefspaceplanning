#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from gpup_gp_node.srv import gpup_transition, gpup_transition_repeat
import math
import numpy as np
from gpup import UncertaintyPropagation
from data_load import data_load
from diffusionMaps import DiffusionMap

np.random.seed(10)

simORreal = 'sim'
discreteORcont = 'discrete'
useDiffusionMaps = False

class Spin_gpup(data_load):

    def __init__(self):
        data_load.__init__(self, simORreal = simORreal, discreteORcont = discreteORcont)

        # Number of NN
        if useDiffusionMaps:
            self.K = 1000
            self.K_manifold = 100
            self.df = DiffusionMap(sigma=1, embedding_dim=3, k = self.K)
        else:
            self.K = 100

        self.opt_count = 0
        self.cur_theta = []

        rospy.Service('/gpup/transition', gpup_transition, self.GetTransition)
        rospy.Service('/gpup/transitionRepeat', gpup_transition_repeat, self.GetTransitionRepeat)
        rospy.init_node('gpup_transition', anonymous=True)
        print('[gpup_transition] Ready.')

        rate = rospy.Rate(15) # 15hz
        while not rospy.is_shutdown():
            rospy.spin()
            # rate.sleep()  

    def predict(self, s_mean, s_var):
        sa = s_mean.reshape(-1,1)
        idx = self.kdt.query(sa.T, k = self.K, return_distance=False)
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

        if useDiffusionMaps:
            X_nn, Y_nn = reduction(sa, X_nn, Y_nn)

        if  not (self.opt_count % 10): # Do not optimize for each prediction
            optimize = True
            theta = None
        else:
            optimize = False
            theta = self.cur_theta

        m = np.zeros(self.state_dim)
        v = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            if i == 0:
                gpup_est = UncertaintyPropagation(X_nn[:,:self.state_dim], Y_nn[:,i], optimize = optimize, theta=theta)
                theta = gpup_est.cov.theta
            else:
                gpup_est = UncertaintyPropagation(X_nn[:,:self.state_dim], Y_nn[:,i], optimize = False, theta=theta)
            m[i], v[i] = gpup_est.predict(sa[:self.state_dim].reshape(1,-1)[0], s_var[:self.state_dim])

        self.cur_theta = theta
        self.opt_count += 1

        return m, np.sqrt(v)

    def reduction(self, sa, X, Y):
        inx = self.df.ReducedClosestSetIndices(sa, X, k_manifold = self.K_manifold)

        return X[inx,:][0], Y[inx,:][0]

    # Predicts the next step by calling the GP class
    def GetTransition(self, req):

        s_mean = np.array(req.mean)
        s_std = np.array(req.std)
        a = np.array(req.action)

        s_mean_normz = self.normz( np.concatenate((s_mean, a), axis=0) )
        s_std_normz = self.normz_change(s_std) # self.normz( s_mean + s_std ) - s_mean_normz[:self.state_dim]

        ds_mean_next_normz, ds_std_next_normz = self.predict(s_mean_normz, s_std_normz**2)

        s_mean_next = self.denormz(s_mean_normz[:self.state_dim] + ds_mean_next_normz)
        s_std_next = np.sqrt(self.denormz_change(ds_std_next_normz)**2 + s_std**2) # self.denormz(s_mean_normz[:self.state_dim] + ds_mean_next_normz + s_std_normz + ds_std_next_normz) - s_mean_next
        
        return {'next_mean': s_mean_next, 'next_std': s_std_next}

    # Predicts the next step by calling the GP class - repeats the same action 'n' times
    def GetTransitionRepeat(self, req):

        s_mean = np.array(req.mean)
        s_std = np.array(req.std)
        a = np.array(req.action)
        n = req.num_repeat

        for _ in range(n):
            s_mean_normz = self.normz( np.concatenate((s_mean, a), axis=0) )
            s_std_normz = self.normz_change(s_std) # self.normz( s_mean + s_std ) - s_mean_normz[:self.state_dim]

            ds_mean_next_normz, ds_std_next_normz = self.predict(s_mean_normz, s_std_normz**2)

            s_mean_next = self.denormz(s_mean_normz[:self.state_dim] + ds_mean_next_normz)
            s_std_next = np.sqrt(self.denormz_change(ds_std_next_normz)**2 + s_std**2) # self.denormz(s_mean_normz[:self.state_dim] + ds_mean_next_normz + s_std_normz + ds_std_next_normz) - s_mean_next

            s_mean = s_mean_next
            s_std = s_std_normz
        
        return {'next_mean': s_mean_next, 'next_std': s_std_next}

        
if __name__ == '__main__':
    try:
        SP = Spin_gpup()
    except rospy.ROSInterruptException:
        pass