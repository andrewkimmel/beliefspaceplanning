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
# probability_threshold = 0.8
plotRegData = False

class Spin_gp(data_load, mean_shift, svm_failure):

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
            self.K = 1
            data_load.__init__(self, simORreal = simORreal, discreteORcont = discreteORcont, K = self.K)

        svm_failure.__init__(self, discrete = (True if discreteORcont=='discrete' else False))
        mean_shift.__init__(self)

        rospy.Service('/inter/transition', batch_transition, self.GetTransition)
        rospy.Service('/inter/transitionOneParticle', one_transition, self.GetTransitionOneParticle)
        rospy.Service('/inter/transitionRepeat', batch_transition_repeat, self.GetTransitionRepeat)
        rospy.Service('/inter/batchSVMcheck', batch_transition, self.batch_svm_check_service)
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
            if V[4]:
                self.precompute_hyperp(K = self.K, K_manifold = self.K_manifold, sigma = sigma, dim = dim)
            print('[gp_transition] Using diffusion maps with dimension %d, K: (%d, %d) and sigma=%f.'%(dim, self.K_manifold, self.K, sigma))
        else:
            useDiffusionMaps = False
            self.K = int(V[1])
            if V[4]:
                self.precompute_hyperp(K = self.K)
            print('[gp_transition] No diffusion maps used, K=%d.'%self.K)

    def inter(self, sa):
        idx = self.kdt.query(sa.reshape(1,-1), k = self.K, return_distance=False)
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

        if useDiffusionMaps:
            X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

        dp = dl = 0
        tp = tl = 0
        for i in range(self.K):
            dp += np.linalg.norm(X_nn[i,:2]-Y_nn[i,6:8])
            dl += np.linalg.norm(X_nn[i,2:4]-Y_nn[i,8:10])
            tp += np.arctan2(Y_nn[i,7]-X_nn[i,1], Y_nn[i,6]-X_nn[i,0])
            tl += np.arctan2(Y_nn[i,1]-X_nn[i,1], Y_nn[i,0]-X_nn[i,0])
        dp /= self.K
        dl /= self.K



        ds_next = np.zeros((self.state_dim,))
        std_next = np.zeros((self.state_dim,))
        for i in range(self.state_dim):
            

            gp_est = GaussianProcess(X_nn[:,:self.state_dim], Y_nn[:,i], optimize = False, theta = self.get_theta(sa))
            mm, vv = gp_est.predict(sa[:self.state_dim])
            ds_next[i] = mm
            std_next[i] = np.sqrt(np.diag(vv))

        s_next = sa[:self.state_dim] + ds_next#np.random.normal(ds_next, std_next)



    # Particles prediction
    def batch_predict(self, SA):
        sa = np.mean(SA, 0)
        idx = self.kdt.query(sa.reshape(1,-1), k = self.K, return_distance=False)
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

        if useDiffusionMaps:
            X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

        dS_next = np.zeros((SA.shape[0], self.state_dim))
        std_next = np.zeros((SA.shape[0], self.state_dim))
        for i in range(self.state_dim):
            gp_est = GaussianProcess(X_nn[:,:self.state_dim], Y_nn[:,i], optimize = False, theta = self.get_theta(sa))
            mm, vv = gp_est.batch_predict(SA[:,:self.state_dim])
            dS_next[:,i] = mm
            std_next[:,i] = np.sqrt(np.diag(vv))

        S_next = SA[:,:self.state_dim] + np.random.normal(dS_next, std_next)

        if plotRegData:
            if np.random.uniform() < 0.1:
                ia = [0, 1]
                ax = plt.subplot(131)
                ax.plot(sa[ia[0]], sa[ia[1]], 'og')
                ax.plot(SA[:,ia[0]], SA[:,ia[1]], '.k')
                ax.plot(X_nn[:,ia[0]], X_nn[:,ia[1]], '.m')
                ax.plot([X_nn[:,ia[0]], X_nn[:,ia[0]]+Y_nn[:,ia[0]]], [X_nn[:,ia[1]], X_nn[:,ia[1]]+Y_nn[:,ia[1]]], '.-b')
                ax.plot([SA[:,ia[0]], S_next[:,ia[0]]], [SA[:,ia[1]], S_next[:,ia[1]]], '-y')
                ax.plot([SA[:,ia[0]], SA[:,ia[0]]+dS_next[:,ia[0]]], [SA[:,ia[1]], SA[:,ia[1]]+dS_next[:,ia[1]]], '-c')
                plt.title('Position')
                ia = [2, 3]
                ax = plt.subplot(132)
                ax.plot(sa[ia[0]], sa[ia[1]], 'og')
                ax.plot(SA[:,ia[0]], SA[:,ia[1]], '.k')
                ax.plot(X_nn[:,ia[0]], X_nn[:,ia[1]], '.m')
                ax.plot([X_nn[:,ia[0]], X_nn[:,ia[0]]+Y_nn[:,ia[0]]], [X_nn[:,ia[1]], X_nn[:,ia[1]]+Y_nn[:,ia[1]]], '.-b')
                ax.plot([SA[:,ia[0]], S_next[:,ia[0]]], [SA[:,ia[1]], S_next[:,ia[1]]], '-y')
                ax.plot([SA[:,ia[0]], SA[:,ia[0]]+dS_next[:,ia[0]]], [SA[:,ia[1]], SA[:,ia[1]]+dS_next[:,ia[1]]], '-c')
                plt.title('Load')
                ia = [4, 5]
                ax = plt.subplot(133)
                ax.plot(sa[ia[0]], sa[ia[1]], 'og')
                ax.plot(SA[:,ia[0]], SA[:,ia[1]], '.k')
                ax.plot(X_nn[:,ia[0]], X_nn[:,ia[1]], '.m')
                ax.plot([X_nn[:,ia[0]], X_nn[:,ia[0]]+Y_nn[:,ia[0]]], [X_nn[:,ia[1]], X_nn[:,ia[1]]+Y_nn[:,ia[1]]], '.-b')
                ax.plot([SA[:,ia[0]], S_next[:,ia[0]]], [SA[:,ia[1]], S_next[:,ia[1]]], '-y')
                ax.plot([SA[:,ia[0]], SA[:,ia[0]]+dS_next[:,ia[0]]], [SA[:,ia[1]], SA[:,ia[1]]+dS_next[:,ia[1]]], '-c')
                plt.title('Velocity')
                plt.show()

        return S_next 

    def one_predict(self, sa):
        idx = self.kdt.query(sa.reshape(1,-1), k = self.K, return_distance=False)
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

        if useDiffusionMaps:
            X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

        ds_next = np.zeros((self.state_dim,))
        std_next = np.zeros((self.state_dim,))
        for i in range(self.state_dim):
            gp_est = GaussianProcess(X_nn[:,:self.state_dim], Y_nn[:,i], optimize = False, theta = self.get_theta(sa))
            mm, vv = gp_est.predict(sa[:self.state_dim])
            ds_next[i] = mm
            std_next[i] = np.sqrt(np.diag(vv))

        s_next = sa[:self.state_dim] + ds_next#np.random.normal(ds_next, std_next)

        if plotRegData:
            # fig = plt.gcf()
            # fig.set_size_inches(24,15)
            # fig.set_figheight(15)
            # fig.set_figwidth(24)
            # print sa
            if np.random.uniform() < 1.1:
                ia = [0, 1]
                ax = plt.subplot(121)
                ax.plot(sa[ia[0]], sa[ia[1]], 'og')
                ax.plot(X_nn[:,ia[0]], X_nn[:,ia[1]], '.m')
                ax.plot([X_nn[:,ia[0]], X_nn[:,ia[0]]+Y_nn[:,ia[0]]], [X_nn[:,ia[1]], X_nn[:,ia[1]]+Y_nn[:,ia[1]]], '.-b')
                ax.plot([sa[ia[0]], s_next[ia[0]]], [sa[ia[1]], s_next[ia[1]]], '-y')
                # ax.plot([SA[:,ia[0]], SA[:,ia[0]]+dS_next[:,ia[0]]], [SA[:,ia[1]], SA[:,ia[1]]+dS_next[:,ia[1]]], '-c')
                plt.title('Position' + str(sa[-2:]))
                ia = [2, 3]
                ax = plt.subplot(122)
                ax.plot(sa[ia[0]], sa[ia[1]], 'og')
                ax.plot(X_nn[:,ia[0]], X_nn[:,ia[1]], '.m')
                ax.plot([X_nn[:,ia[0]], X_nn[:,ia[0]]+Y_nn[:,ia[0]]], [X_nn[:,ia[1]], X_nn[:,ia[1]]+Y_nn[:,ia[1]]], '.-b')
                ax.plot([sa[ia[0]], s_next[ia[0]]], [sa[ia[1]], s_next[ia[1]]], '-y')
                # ax.plot([SA[:,ia[0]], SA[:,ia[0]]+dS_next[:,ia[0]]], [SA[:,ia[1]], SA[:,ia[1]]+dS_next[:,ia[1]]], '-c')
                plt.title('Load - DF')
                # ia = [4, 5]
                # ax = plt.subplot(133)
                # ax.plot(sa[ia[0]], sa[ia[1]], 'og')
                # ax.plot(SA[:,ia[0]], SA[:,ia[1]], '.k')
                # ax.plot(X_nn[:,ia[0]], X_nn[:,ia[1]], '.m')
                # ax.plot([X_nn[:,ia[0]], X_nn[:,ia[0]]+Y_nn[:,ia[0]]], [X_nn[:,ia[1]], X_nn[:,ia[1]]+Y_nn[:,ia[1]]], '.-b')
                # ax.plot([SA[:,ia[0]], S_next[:,ia[0]]], [SA[:,ia[1]], S_next[:,ia[1]]], '-y')
                # ax.plot([SA[:,ia[0]], SA[:,ia[0]]+dS_next[:,ia[0]]], [SA[:,ia[1]], SA[:,ia[1]]+dS_next[:,ia[1]]], '-c')
                # plt.title('Velocity')
                # plt.show()
                plt.savefig('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/temp2/pic_' + str(rospy.get_time()) + '.png')
                plt.clf()

        return s_next 

    def reduction(self, sa, X, Y):
        inx = self.df.ReducedClosestSetIndices(sa, X, k_manifold = self.K_manifold)

        return X[inx,:][0], Y[inx,:][0]

    def batch_propa(self, S, a):
        SA = np.concatenate((S, np.tile(a, (S.shape[0],1))), axis=1)

        SA = self.normz_batch( SA )    
        SA_normz = self.batch_predict(SA)
        S_next = self.denormz_batch( SA_normz )

        return S_next

    def batch_svm_check(self, S, a):
        failed_inx = []
        for i in range(S.shape[0]):
            p, _ = self.probability(S[i,:], a) # Probability of failure
            prob_fail = np.random.uniform(0,1)
            if prob_fail <= p:
                failed_inx.append(i)

        return failed_inx

    def batch_svm_check_service(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)

        failed_inx = []
        for i in range(S.shape[0]):
            p, _ = self.probability(S[i,:], a) # Probability of failure
            prob_fail = np.random.uniform(0,1)
            if prob_fail <= p:
                failed_inx.append(i)

        node_probability = 1.0 - float(len(failed_inx))/float(S.shape[0])

        return {'node_probability': node_probability}

    # Predicts the next step by calling the GP class
    def GetTransition(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)

        # Check which particles failed
        failed_inx = self.batch_svm_check(S, a)
        # node_probability = 1 - len(failed_inx)/S.shape[0]
        node_probability = 1.0 - float(len(failed_inx))/float(S.shape[0])

        # print node_probability

        # Remove failed particles by duplicating good ones
        if len(failed_inx):
            good_inx = np.delete( np.array(range(S.shape[0])), failed_inx )
            if len(good_inx) == 0: # All particles failed
                S_next = []
                mean = [0,0]
                return {'next_states': S_next, 'mean_shift': mean, 'node_probability': node_probability}

            dup_inx = np.random.choice(len(good_inx), size=len(failed_inx), replace=True)
            S[failed_inx, :] = S[dup_inx,:]

        # Propagate
        S_next = self.batch_propa(S, a)

        mean = self.get_mean_shift(S_next)
        
        return {'next_states': S_next.reshape((-1,)), 'mean_shift': mean, 'node_probability': node_probability}

    # Predicts the next step by calling the GP class - repeats the same action 'n' times
    def GetTransitionRepeat(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)
        n = req.num_repeat

        for _ in range(n):
            S_next = self.batch_propa(S, a)
            S = S_next
        
        mean = self.get_mean_shift(S_next)
        
        return {'next_states': S_next.reshape((-1,)), 'mean_shift': mean}

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
        sa_normz = self.one_predict(sa)
        s_next = self.denormz( sa_normz )

        return {'next_state': s_next, 'node_probability': node_probability}

if __name__ == '__main__':
    try:
        SP = Spin_gp()
    except rospy.ROSInterruptException:
        pass