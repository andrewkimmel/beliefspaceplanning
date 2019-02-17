#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from gpup_gp_node.srv import batch_transition, batch_transition_repeat, one_transition, setk
import math
import numpy as np
from data_load import data_load
from svm_class import svm_failure
from dr_diffusionmaps import DiffusionMap
from mean_shift import mean_shift
import matplotlib.pyplot as plt

# np.random.seed(10)

simORreal = 'sim'
discreteORcont = 'discrete'
useDiffusionMaps = False
probability_threshold = 0.65
plotRegData = False

class Spin_inter(data_load, mean_shift, svm_failure):

    def __init__(self):
        # Number of NN
        if useDiffusionMaps:
            dim = 2
            self.K = 500
            self.K_manifold = 50
            sigma = 5
            self.df = DiffusionMap(sigma=sigma, embedding_dim=dim)
            # self.df = DiffusionMap(sigma=10, embedding_dim=dim, k = self.K)
            print('[gp_transition] Using diffusion maps with dimension %d, K: (%d, %d) and sigma=%f.'%(dim, self.K_manifold, self.K, sigma))
            data_load.__init__(self, simORreal = simORreal, discreteORcont = discreteORcont, K = self.K, K_manifold = self.K_manifold, sigma=sigma, dim = dim)
        else:
            self.K = 50
            data_load.__init__(self, simORreal = simORreal, discreteORcont = discreteORcont, K = self.K)

        svm_failure.__init__(self, discrete = (True if discreteORcont=='discrete' else False))
        mean_shift.__init__(self)

        # rospy.Service('/inter/transition', batch_transition, self.GetTransition)
        rospy.Service('/inter/transitionOneParticle', one_transition, self.GetTransitionOneParticle)
        # rospy.Service('/inter/transitionRepeat', batch_transition_repeat, self.GetTransitionRepeat)
        # rospy.Service('/inter/batchSVMcheck', batch_transition, self.batch_svm_check_service)
        rospy.Service('/inter/set_K', setk, self.setK)
        rospy.init_node('inter_transition', anonymous=True)
        print('[inter_transition] Ready.')            

        rate = rospy.Rate(15) # 15hz
        while not rospy.is_shutdown():
            rospy.spin()
            # rate.sleep()  

    def setK(self, msg):
        V = np.array(msg.data)

        if V[0]  < self.state_dim:
            useDiffusionMaps = True
            dim = int(V[0])
            self.K_manifold = int(V[1])
            self.K = int(V[2])
            sigma = V[3]
            self.df = DiffusionMap(sigma=sigma, embedding_dim=dim)
            print('[gp_transition] Using diffusion maps with dimension %d, K: (%d, %d) and sigma=%f.'%(dim, self.K_manifold, self.K, sigma))
        else:
            useDiffusionMaps = False
            self.K = int(V[1])
            print('[gp_transition] No diffusion maps used, K=%d.'%self.K)

    def inter(self, sa):
        idx = self.kdt.query(sa.reshape(1,-1), k = self.K, return_distance=False)
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

        if useDiffusionMaps:
            X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

        W = []
        for i in range(self.K):
            W.append( self.cov(sa[:self.state_dim], X_nn[i, :self.state_dim]) )
        W = np.array(W)
        W /= np.sum(W)

        Yw = np.multiply(Y_nn, np.tile(W.reshape((-1,1)), (1,self.state_dim)))

        d_mean = np.sum(Yw, axis=0)
        # d_std = np.std(Yw, axis=0)
        
        s_next = sa[:self.state_dim] + d_mean

        # ia = [0, 1]
        # ax = plt.subplot(111)
        # ax.plot(sa[ia[0]], sa[ia[1]], 'og')
        # ax.plot(X_nn[:,ia[0]], X_nn[:,ia[1]], '.m')
        # ax.plot([X_nn[:,ia[0]], X_nn[:,ia[0]]+Y_nn[:,ia[0]]], [X_nn[:,ia[1]], X_nn[:,ia[1]]+Y_nn[:,ia[1]]], '.-b')
        # ax.plot([sa[ia[0]], s_next[ia[0]]], [sa[ia[1]], s_next[ia[1]]], '-y')
        # plt.show()
        # exit(1)


        # s_next = sa[:self.state_dim] + np.random.normal(ds_next, std_next)

        return s_next


    def reduction(self, sa, X, Y):
        inx = self.df.ReducedClosestSetIndices(sa, X, k_manifold = self.K_manifold)

        return X[inx,:][0], Y[inx,:][0]

    # Predicts the next step by calling the GP class
    def GetTransitionOneParticle(self, req):

        s = np.array(req.state)
        a = np.array(req.action)

        # Check which particles failed
        p, _ = self.probability(s, a)
        node_probability = 1 - p

        # Propagate
        sa = np.concatenate((s, a), axis=0)
        sa = self.normz( sa )    
        sa_normz = self.inter(sa)
        s_next = self.denormz( sa_normz )

        return {'next_state': s_next, 'node_probability': node_probability}

    def cov(self,xi,xj):
        # Computes a scalar covariance of two samples

        v = 1.
        w = 0.001 * np.ones((self.state_dim,1))

        diff = xi - xj
        W = 1. / w

        #slighly dirty hack to determine whether i==j
        return v * np.exp(-0.5 * (np.dot(diff.T, W* diff)))

if __name__ == '__main__':
    try:
        SP = Spin_inter()
    except rospy.ROSInterruptException:
        pass