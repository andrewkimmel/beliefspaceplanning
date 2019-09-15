#!/usr/bin/env python

import rospy
from gpup_gp_node.srv import batch_transition, one_transition
from sim_nn_node.srv import critic_seq
import numpy as np
from svm_class import svm_failure
import pickle

from predict_nn import predict_nn

from sklearn.neighbors import KDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

CRITIC = True

class Spin_predict(predict_nn, svm_failure):

    state_dim = 4
    OBS = True

    def __init__(self):
        predict_nn.__init__(self)
        svm_failure.__init__(self, simORreal = 'sim_cyl19', discrete = False)

        rospy.Service('/nn/transition', batch_transition, self.GetTransition)
        rospy.Service('/nn/transitionOneParticle', one_transition, self.GetTransitionOneParticle)
        if CRITIC:
            rospy.Service('/nn/critic_seq', critic_seq, self.GetCritic)

        rospy.init_node('predict', anonymous=True)

        if self.OBS:
            with open('/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/obs_19.pkl', 'r') as f: 
                self.Obs = pickle.load(f)

        if CRITIC:
            self.K = 3
            with open('/home/juntao/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/data_P40_sakh_v2.pkl', 'rb') as f: 
                self.O, self.E = pickle.load(f)
            if 1:
                self.kdt = KDTree(self.O, leaf_size=100, metric='euclidean')
            else:
                with open('/home/juntao/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/kdt_P40_sakh_v2.pkl', 'rb') as f: 
                    self.kdt = pickle.load(f)
            self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
            print('[nn_predict_node] Critic loaded.')

        print('[nn_predict_node] Ready to predict...')
        rospy.spin()

    def batch_svm_check(self, S, a):
        failed_inx = []
        for i in range(S.shape[0]):
            p = self.probability(S[i,:], a) # Probability of failure
            prob_fail = np.random.uniform(0,1)
            if prob_fail <= p:
                failed_inx.append(i)

        return failed_inx

    # Predicts the next step by calling the GP class
    def GetTransition(self, req):

        # print "In NN"

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)

        collision_probability = 0.0

        if (len(S) == 1):
            st = rospy.get_time()
            p = self.probability(S[0,:], a)

            node_probability = 1.0 - p
            sa = np.concatenate((S[0,:],a), axis=0)
            s_next = self.predict(sa)

            collision_probability = 1.0 if (self.OBS and self.obstacle_check(s_next)) else 0.0

            # print "Out NN", s_next

            return {'next_states': s_next, 'mean_shift': s_next, 'node_probability': node_probability, 'collision_probability': collision_probability}
        else:       
            # Check which particles failed
            failed_inx = self.batch_svm_check(S, a)
            try:
                node_probability = 1.0 - float(len(failed_inx))/float(S.shape[0])
            except:
                S_next = []
                mean = [0,0]
                return {'next_states': S_next, 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': np.array([0.,0.]), 'collision_probability': 1.0}

            # Remove failed particles by duplicating good ones
            bad_action = np.array([0.,0.])
            if len(failed_inx):
                good_inx = np.delete( np.array(range(S.shape[0])), failed_inx )
                if len(good_inx) == 0: # All particles failed
                    S_next = []
                    mean = [0,0]
                    return {'next_states': S_next, 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': np.array([0.,0.]), 'collision_probability': 1.0}

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
            SA = np.concatenate((S, np.tile(a, (S.shape[0],1))), axis=1)
            S_next = self.predict(SA)

            mean = np.mean(S_next, 0) #self.get_mean_shift(S_next)
            return {'next_states': S_next.reshape((-1,)), 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': bad_action, 'collision_probability': collision_probability}

    def GetTransitionOneParticle(self, req):

        s = np.array(req.state)
        a = np.array(req.action)

        # Check which particles failed
        p = self.probability(s, a)
        node_probability = 1.0 - p

        # Propagate
        sa = np.concatenate((s, a), axis=0)
        s_next = self.predict(sa) 

        # print(self.time_nn / self.num_checks_nn) 

        return {'next_state': s_next, 'node_probability': node_probability}

    def obstacle_check(self, s):
        f = 3.0 #2.0 # inflate

        for o in self.Obs:
            if np.linalg.norm(s[:2]-o[:2]) <= f * o[2]:
                # print "right obstacle collision"
                return True
        return False

    def GetCritic(self, req):
        # print "In critic"

        s = np.array(req.state)
        a = np.array(req.future_action)
        n = req.steps
        Apr = np.array(req.seq)

        # sa = np.concatenate((s, a, np.array([n]), Apr.reshape((-1))), axis=0)
        sa = np.concatenate((s, a, Apr.reshape((-1))), axis=0)
        # sa = np.append(sa, n)

        idx = self.kdt.query(sa.reshape(1,-1), k = self.K, return_distance=False)
        O_nn = self.O[idx,:].reshape(self.K, -1)
        E_nn = self.E[idx].reshape(self.K, 1)

        gpr = GaussianProcessRegressor(kernel=self.kernel).fit(O_nn, E_nn)
        e, _ = gpr.predict(sa.reshape(1, -1), return_std=True)
        # print "Out critic"
    
        return {'err': e}



if __name__ == '__main__':
    
    try:
        SP = Spin_predict()
    except rospy.ROSInterruptException:
        pass