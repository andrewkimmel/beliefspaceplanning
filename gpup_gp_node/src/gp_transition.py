#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
from gpup_gp_node.srv import batch_transition, batch_transition_repeat, one_transition, setk
import math
import numpy as np
from gp import GaussianProcess
from data_load import data_load
from svm_class import svm_failure
from diffusionMaps import DiffusionMap
# from dr_diffusionmaps import DiffusionMap
from spectralEmbed import spectralEmbed
from mean_shift import mean_shift
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors

# np.random.seed(10)

simORreal = 'acrobot'
discreteORcont = 'cont'
useDiffusionMaps = False
probability_threshold = 0.65
plotRegData = False
diffORspec = 'diff'

class Spin_gp(data_load, mean_shift):#, svm_failure):

    OBS = False

    def __init__(self):
        # Number of NN
        if useDiffusionMaps:
            dim = 3
            self.K = 1000
            self.K_manifold = 100
            sigma = 5
            if diffORspec == 'diff':
                # self.df = DiffusionMap(sigma=sigma, embedding_dim=dim)
                self.df = DiffusionMap(sigma=10, embedding_dim=dim, k = self.K)
                print('[gp_transition] Using diffusion maps with dimension %d, K: (%d, %d) and sigma=%f.'%(dim, self.K_manifold, self.K, sigma))
            else:
                self.embedding = spectralEmbed(embedding_dim=dim) 
                print('[gp_transition] Using spectral embedding with dimension %d.'%(dim))
            data_load.__init__(self, simORreal = simORreal, discreteORcont = discreteORcont, K = self.K, K_manifold = self.K_manifold, sigma=sigma, dim = dim, dr = 'diff')
        else:
            self.K = 1000
            print('[gp_transition] No diffusion maps used, K=%d.'%self.K)
            data_load.__init__(self, simORreal = simORreal, discreteORcont = discreteORcont, K = self.K, dr = 'spec')

        # svm_failure.__init__(self, discrete = (True if discreteORcont=='discrete' else False))
        mean_shift.__init__(self)

        rospy.Service('/gp/transition', batch_transition, self.GetTransition)
        rospy.Service('/gp/transitionOneParticle', one_transition, self.GetTransitionOneParticle)
        rospy.Service('/gp/transitionRepeat', batch_transition_repeat, self.GetTransitionRepeat)
        rospy.Service('/gp/batchSVMcheck', batch_transition, self.batch_svm_check_service)
        rospy.Service('/gp/set_K', setk, self.setK)
        rospy.init_node('gp_transition', anonymous=True)
        print('[gp_transition] Ready.')            

        rospy.spin()

    def setK(self, msg):
        V = np.array(msg.data)

        if V[0]  < self.state_dim:
            useDiffusionMaps = True
            dim = int(V[0])
            self.K_manifold = int(V[1])
            self.K = int(V[2])
            if diffORspec == 'diff':
                sigma = V[3]
                self.df = DiffusionMap(sigma=sigma, embedding_dim=dim)
                if V[4]:
                    self.dr = 'diff'
                    self.precompute_hyperp(K = self.K, K_manifold = self.K_manifold, sigma = sigma, dim = dim)
                print('[gp_transition] Using diffusion maps with dimension %d, K: (%d, %d) and sigma=%f.'%(dim, self.K_manifold, self.K, sigma))
            elif diffORspec == 'spec':
                self.embedding = spectralEmbed(embedding_dim=dim)
                if V[4]:
                    self.dr = 'spec'
                    self.precompute_hyperp(K = self.K, K_manifold = self.K_manifold, dim = dim) 
                print('[gp_transition] Using spectral embedding with dimension %d.'%(dim))
        else:
            useDiffusionMaps = False
            self.K = int(V[1])
            if V[4]:
                self.precompute_hyperp(K = self.K)
            print('[gp_transition] No diffusion maps used, K=%d.'%self.K)

        # return EmptyResponse()

    # Particles prediction
    def batch_predict(self, SA):
        # If the particles are seperated, move them to one cluster via wraping
        bi = [False, False]
        for i in range(np.minimum(25, SA.shape[0])):
            x1 = SA[np.random.randint(SA.shape[0]), :2]
            x2 = SA[np.random.randint(SA.shape[0]), :2]
            if np.abs(x1[0]-x2[0]) > 0.97:
                bi[0] = True
            if np.abs(x1[1]-x2[1]) > 0.97:
                bi[1] = True
        for i in range(2):
            if bi[i]:
                C = 0
                for sa in SA:
                    C += 1 if sa[i] > 0.5 else 0 # Move the minority particles
                for sa in SA:
                    if C > SA.shape[0]:
                        sa[i] += 1. if sa[i] < 0.5 else 0.0
                    else:
                        sa[i] -= 1. if sa[i] > 0.5 else 0.0                

        sa = np.mean(SA, 0)
        # idx = self.kdt.kneighbors(np.copy(sa).reshape(1,-1), n_neighbors = self.K, return_distance=False)
        idx = self.kdt.query(np.copy(sa).reshape(1,-1), k = self.K)[0]
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

        # If the neighbors are seperated, move them to one cluster via wraping
        bi = [False, False]
        for i in range(np.minimum(25, X_nn.shape[0])):
            x1 = X_nn[np.random.randint(X_nn.shape[0]), :2]
            x2 = X_nn[np.random.randint(X_nn.shape[0]), :2]
            if np.abs(x1[0]-x2[0]) > 0.97:
                bi[0] = True
            if np.abs(x1[1]-x2[1]) > 0.97:
                bi[1] = True
        for i in range(2):
            if bi[i]:
                for x in X_nn:
                    if sa[i] > 0.5:
                        x[i] += 1. if x[i] < 0.5 else 0
                    else:
                        x[i] -= 1. if x[i] > 0.5 else 0 

        if useDiffusionMaps:
            X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

        Theta = self.get_theta(sa) # Get hyper-parameters for this query point

        dS_next = np.zeros((SA.shape[0], self.state_dim))
        std_next = np.zeros((SA.shape[0], self.state_dim))
        for i in range(self.state_dim):
            gp_est = GaussianProcess(X_nn[:,:self.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i], algorithm = 'Matlab')
            mm, vv = gp_est.batch_predict(SA[:,:self.state_action_dim])
            dS_next[:,i] = mm
            std_next[:,i] = np.sqrt(np.diag(vv))

        S_next = SA[:,:self.state_dim] + np.random.normal(dS_next, std_next)
        for s_next in S_next:
            for i in range(2):
                s_next[i] += 1.0 if s_next[i] < 1.0 else 0.0
                s_next[i] -= 1.0 if s_next[i] > 1.0 else 0.0 

        return S_next

    # Particles prediction
    def batch_predict_iterative(self, SA):

        S_next = []
        while SA.shape[0]:
            sa = np.copy(SA[np.random.randint(SA.shape[0]), :])
            D = self.kdt.query(sa.reshape(1,-1), k = self.K)
            idx = D[0]
            r = np.max(D[1])*1.1
            X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
            Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)

            neigh = NearestNeighbors(radius=r)
            neigh.fit(SA)
            idx_local = neigh.radius_neighbors(sa.reshape(1,-1),return_distance=False)[0]
            SA_local = np.copy(SA[idx_local, :])
            SA = np.delete(SA, idx_local, axis = 0)

            # If the neighbors are seperated, move them to one cluster via wraping
            bi = [False, False]
            for i in range(np.minimum(25, X_nn.shape[0])):
                x1 = X_nn[np.random.randint(X_nn.shape[0]), :2]
                x2 = X_nn[np.random.randint(X_nn.shape[0]), :2]
                if np.abs(x1[0]-x2[0]) > 0.97:
                    bi[0] = True
                if np.abs(x1[1]-x2[1]) > 0.97:
                    bi[1] = True
            for i in range(2):
                if bi[i]:
                    for x in X_nn:
                        if sa[i] > 0.5:
                            x[i] += 1. if x[i] < 0.5 else 0
                        else:
                            x[i] -= 1. if x[i] > 0.5 else 0 

            if useDiffusionMaps:
                X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

            Theta = self.get_theta(sa) # Get hyper-parameters for this query point

            dS_next = np.zeros((SA_local.shape[0], self.state_dim))
            std_next = np.zeros((SA_local.shape[0], self.state_dim))
            for i in range(self.state_dim):
                gp_est = GaussianProcess(X_nn[:,:self.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i], algorithm = 'Matlab')
                mm, vv = gp_est.batch_predict(SA_local[:,:self.state_action_dim])
                dS_next[:,i] = mm
                std_next[:,i] = np.sqrt(np.diag(vv))

            S_next_local = SA_local[:,:self.state_dim] + np.random.normal(dS_next, std_next)
            for s_next in S_next_local:
                for i in range(2):
                    s_next[i] += 1.0 if s_next[i] < 1.0 else 0.0
                    s_next[i] -= 1.0 if s_next[i] > 1.0 else 0.0 

            for s in S_next_local:
                S_next.append(s)

        return np.array(S_next)

    def one_predict(self, sa):
        st = time.time()
        # idx = self.kdt.kneighbors(np.copy(sa).reshape(1,-1), n_neighbors = self.K, return_distance=False)
        idx = self.kdt.query(np.copy(sa).reshape(1,-1), k = self.K)[0]
        X_nn = self.Xtrain[idx,:].reshape(self.K, self.state_action_dim)
        Y_nn = self.Ytrain[idx,:].reshape(self.K, self.state_dim)
        tnn = time.time() - st

        # If the neighbors are seperated, move them to one cluster via wraping
        bi = [False, False]
        for i in range(np.minimum(25, X_nn.shape[0])):
            x1 = X_nn[np.random.randint(X_nn.shape[0]), :2]
            x2 = X_nn[np.random.randint(X_nn.shape[0]), :2]
            if np.abs(x1[0]-x2[0]) > 0.97:
                bi[0] = True
            if np.abs(x1[1]-x2[1]) > 0.97:
                bi[1] = True
        for i in range(2):
            if bi[i]:
                for x in X_nn:
                    if sa[i] > 0.5:
                        x[i] += 1. if x[i] < 0.5 else 0
                    else:
                        x[i] -= 1. if x[i] > 0.5 else 0 

        if useDiffusionMaps:
            X_nn, Y_nn = self.reduction(sa, X_nn, Y_nn)

        Theta = self.get_theta(sa) # Get hyper-parameters for this query point

        ds_next = np.zeros((self.state_dim,))
        std_next = np.zeros((self.state_dim,))
        tgp = 0
        tp = 0
        for i in range(self.state_dim):
            st = time.time()
            gp_est = GaussianProcess(X_nn[:,:self.state_action_dim], Y_nn[:,i], optimize = False, theta = Theta[i], algorithm = 'Matlab')
            tgp += time.time() - st
            st = time.time()
            mm, vv = gp_est.predict(sa[:self.state_action_dim])
            tp += time.time() - st
            ds_next[i] = mm
            std_next[i] = np.sqrt(vv)

        # print tgp / self.state_dim, tp / self.state_dim, tnn

        s_next = sa[:self.state_dim] + np.random.normal(ds_next, std_next)
        for i in range(2):
            s_next[i] += 1.0 if s_next[i] < 1.0 else 0.0
            s_next[i] -= 1.0 if s_next[i] > 1.0 else 0.0        

        return s_next

    def reduction(self, sa, X, Y):
        if diffORspec == 'diff':
            inx = self.df.ReducedClosestSetIndices(sa, X, k_manifold = self.K_manifold)
        elif diffORspec == 'spec':
            inx = self.embedding.ReducedClosestSetIndices(sa, X, k_manifold = self.K_manifold)

        return X[inx,:][0], Y[inx,:][0]

    def batch_propa(self, S, a):
        SA = np.concatenate((S, np.tile(a, (S.shape[0],1))), axis=1)

        SA = self.normz_batch( SA )    
        # SA_normz = self.batch_predict(SA)
        SA_normz = self.batch_predict_iterative(SA)
        S_next = self.denormz_batch( SA_normz )

        return S_next

    def batch_svm_check(self, S, a):
        failed_inx = []
        # for i in range(S.shape[0]):
        #     p, _ = self.probability(S[i,:], a) # Probability of failure
        #     prob_fail = np.random.uniform(0,1)
        #     if prob_fail <= p:
        #         failed_inx.append(i)

        return failed_inx

    def batch_svm_check_service(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)

        failed_inx = []
        # for i in range(S.shape[0]):
        #     p, _ = self.probability(S[i,:], a) # Probability of failure
        #     prob_fail = np.random.uniform(0,1)
        #     if prob_fail <= p:
        #         failed_inx.append(i)

        node_probability = 1.0 - float(len(failed_inx))/float(S.shape[0])

        return {'node_probability': node_probability}

    # Predicts the next step by calling the GP class
    def GetTransition(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)

        if (len(S) == 1):
            # p, _ = self.probability(S[0,:],a)
            node_probability = 1.0# - p
            sa = np.concatenate((S[0,:],a), axis=0)
            sa = self.normz(sa)
            sa_normz = self.one_predict(sa)
            s_next = self.denormz(sa_normz)

            if self.OBS and self.obstacle_check(s_next):
                node_probability = 0.0

            return {'next_states': s_next, 'mean_shift': s_next, 'node_probability': node_probability}
        else:       

            # Check which particles failed
            failed_inx = self.batch_svm_check(S, a)
            node_probability = 1.0 - float(len(failed_inx))/float(S.shape[0])

            # Remove failed particles by duplicating good ones
            bad_action = np.array([0.,0.])
            if len(failed_inx):
                good_inx = np.delete( np.array(range(S.shape[0])), failed_inx )
                if len(good_inx) == 0: # All particles failed
                    S_next = []
                    mean = [0,0]
                    return {'next_states': S_next, 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': np.array([0.,0.])}

                # Find main direction of fail
                S_failed_mean = np.mean(S[failed_inx, :], axis=0)
                S_mean = np.mean(S, axis=0)
                ang = np.rad2deg(np.arctan2(S_failed_mean[1]-S_mean[1], S_failed_mean[0]-S_mean[0]))
                if ang <= 45. and ang >= -45.:
                    bad_action = np.array([1.,-1.])
                elif ang >= 135. or ang <= -135.:
                    bad_action = np.array([-1.,1.])
                elif ang > 45. and ang < 135.:
                    bad_action = np.array([1.,1.])
                elif ang < -45. and ang > -135.:
                    bad_action = np.array([-1.,-1.])

                dup_inx = good_inx[np.random.choice(len(good_inx), size=len(failed_inx), replace=True)]
                S[failed_inx, :] = S[dup_inx,:]

            # Propagate
            S_next = self.batch_propa(S, a)

            if self.OBS:
                # print "Checking obstacles..."
                failed_inx = []
                good_inx = []
                for i in range(S_next.shape[0]):
                    if self.obstacle_check(S_next[i,:]):
                        failed_inx.append(i)
                node_probability2 = 1.0 - float(len(failed_inx))/float(S.shape[0])
                node_probability = min(node_probability, node_probability2)

                if len(failed_inx):
                    good_inx = np.delete( np.array(range(S_next.shape[0])), failed_inx )
                    if len(good_inx) == 0: # All particles failed
                        S_next = []
                        mean = [0,0]
                        return {'next_states': S_next, 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': np.array([0.,0.])}

                    # Find main direction of fail
                    S_next_failed_mean = np.mean(S_next[failed_inx, :], axis=0)
                    S_next_mean = np.mean(S_next, axis=0)
                    ang = np.rad2deg(np.arctan2(S_next_failed_mean[1]-S_next_mean[1], S_next_failed_mean[0]-S_next_mean[0]))
                    if ang <= 45. and ang >= -45.:
                        bad_action = np.array([1.,-1.])
                    elif ang >= 135. or ang <= -135.:
                        bad_action = np.array([-1.,1.])
                    elif ang > 45. and ang < 135.:
                        bad_action = np.array([1.,1.])
                    elif ang < -45. and ang > -135.:
                        bad_action = np.array([-1.,-1.])

                    dup_inx = good_inx[np.random.choice(len(good_inx), size=len(failed_inx), replace=True)]
                    S_next[failed_inx, :] = S_next[dup_inx,:]

            mean = self.get_mean_shift(S_next)
            
            return {'next_states': S_next.reshape((-1,)), 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': bad_action}

    def obstacle_check(self, s):
        # Obs1 = np.array([42, 90, 12.])
        # Obs2 = np.array([-45, 101, 7.])
        # f = 1.15 # inflate
        Obs1 = np.array([33, 110, 4.]) # Right
        Obs2 = np.array([-27, 118, 2.5]) # Left
        f = 1.75 # inflate

        if np.linalg.norm(s[:2]-Obs1[:2]) <= f * Obs1[2]:
            # print "right obstacle collision"
            return True
        elif np.linalg.norm(s[:2]-Obs2[:2]) <= f * Obs2[2]:
            # print "left obstacle collision", s[:2], Obs2[:2]
            return True
        else:
            return False

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
        # p, _ = self.probability(s, a)
        node_probability = 1.0# - p

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