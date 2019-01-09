#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from gpup_gp_node.srv import batch_transition, batch_transition_repeat
import math
import numpy as np
from gp import GaussianProcess
from data_load import data_load
from diffusionMaps import DiffusionMap

np.random.seed(10)

simORreal = 'sim'
discreteORcont = 'discrete'
useDiffusionMaps = False

class Spin_gp(data_load):

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

        rospy.Service('/gp/transition', batch_transition, self.GetTransition)
        rospy.Service('/gp/transitionRepeat', batch_transition_repeat, self.GetTransitionRepeat)
        rospy.init_node('gp_transition', anonymous=True)
        print('[gp_transition] Ready.')

        rate = rospy.Rate(15) # 15hz
        while not rospy.is_shutdown():
            rospy.spin()
            # rate.sleep()  

    # Particles prediction
    def batch_predict(self, SA):
        sa = np.mean(SA, 0)
        idx = self.kdt.query(sa.reshape(1,-1), k = self.K, return_distance=False)
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

        if useDiffusionMaps:
            X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

        if  not (self.opt_count % 10): # Do not optimize for each prediction
            optimize = True
            theta = None
        else:
            optimize = False
            theta = self.cur_theta

        dS_next = np.zeros((SA.shape[0], self.state_dim))
        std_next = np.zeros((SA.shape[0], self.state_dim))
        for i in range(self.state_dim):
            if i == 0:
                gp_est = GaussianProcess(X_nn[:,:self.state_dim], Y_nn[:,i], optimize = optimize, theta=theta)
                theta = gp_est.cov.theta
            else:
                gp_est = GaussianProcess(X_nn[:,:self.state_dim], Y_nn[:,i], optimize = False, theta=theta)
            mm, vv = gp_est.batch_predict(SA[:,:self.state_dim])
            dS_next[:,i] = mm
            std_next[:,i] = np.sqrt(np.diag(vv))

        self.cur_theta = theta
        self.opt_count += 1

        S_next = SA[:,:self.state_dim] + np.random.normal(dS_next, std_next)

        return S_next    

    def reduction(self, sa, X, Y):
        inx = self.df.ReducedClosestSetIndices(sa, X, k_manifold = self.K_manifold)

        return X[inx,:][0], Y[inx,:][0]

    # Predicts the next step by calling the GP class
    def GetTransition(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)
        SA = np.concatenate((S, np.tile(a, (S.shape[0],1))), axis=1)

        SA = self.normz_batch( SA )    
        SA_normz = self.batch_predict(SA)
        S_next = self.denormz_batch( SA_normz )
        
        return {'next_states': S_next.reshape((-1,))}

    # Predicts the next step by calling the GP class - repeats the same action 'n' times
    def GetTransitionRepeat(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)
        n = req.num_repeat

        for _ in range(n):
            SA = np.concatenate((S, np.tile(a, (S.shape[0],1))), axis=1)

            SA = self.normz_batch( SA )    
            SA_normz = self.batch_predict(SA)
            S_next = self.denormz_batch( SA_normz )

            S = S_next
        
        return {'next_states': S_next.reshape((-1,))}

        
if __name__ == '__main__':
    try:
        SP = Spin_gp()
    except rospy.ROSInterruptException:
        pass