#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Int16
from std_srvs.srv import SetBool, Empty, EmptyResponse
import math
import numpy as np
import matplotlib.pyplot as plt
from predict_nn import predict_nn
from svm_class import svm_failure


import sys
sys.path.insert(0, '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/')
from mean_shift import mean_shift
from gpup_gp_node.srv import batch_transition, batch_transition_repeat, one_transition, setk

# np.random.seed(10)

simORreal = 'sim'
discreteORcont = 'discrete'
probability_threshold = 0.65


class Spin_nn(predict_nn, mean_shift, svm_failure):

    OBS = True

    def __init__(self):
        
        predict_nn.__init__(self)
        svm_failure.__init__(self, discrete = (True if discreteORcont=='discrete' else False))
        mean_shift.__init__(self)

        rospy.Service('/nn/transition', batch_transition, self.GetTransition)
        rospy.Service('/nn/transitionOneParticle', one_transition, self.GetTransitionOneParticle)
        rospy.Service('/nn/transitionRepeat', batch_transition_repeat, self.GetTransitionRepeat)
        rospy.Service('/nn/batchSVMcheck', batch_transition, self.batch_svm_check_service)
        rospy.init_node('nn_transition', anonymous=True)
        print('[nn_transition] Ready.')  

        self.time_svm = 0.
        self.num_checks_svm = 0     
        self.time_nn = 0.
        self.num_checks_nn = 0
        self.time_bnn = 0.
        self.num_checks_bnn = 0        

        rospy.spin()

    def batch_svm_check(self, S, a):
        failed_inx = []
        for i in range(S.shape[0]):
            st = rospy.get_time()
            p = self.probability(S[i,:], a) # Probability of failure
            self.time_svm += rospy.get_time() - st
            self.num_checks_svm += 1
            prob_fail = np.random.uniform(0,1)
            if prob_fail <= p:
                failed_inx.append(i)

        return failed_inx

    def batch_svm_check_service(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)

        failed_inx = []
        for i in range(S.shape[0]):
            p = self.probability(S[i,:], a) # Probability of failure
            prob_fail = np.random.uniform(0,1)
            if prob_fail <= p:
                failed_inx.append(i)

        node_probability = 1.0 - float(len(failed_inx))/float(S.shape[0])

        return {'node_probability': node_probability}

    # Predicts the next step by calling the GP class
    def GetTransition(self, req):

        S = np.array(req.states).reshape(-1, self.state_dim)
        a = np.array(req.action)

        if (len(S) == 1):
            st = rospy.get_time()
            p = self.probability(S[0,:], a)
            self.time_svm += rospy.get_time() - st
            self.num_checks_svm += 1

            print("------")
            print(S[0,:], a)

            node_probability = 1.0 - p
            sa = np.concatenate((S[0,:],a), axis=0)
            st = rospy.get_time()
            s_next = self.predict(sa)
            self.time_nn += rospy.get_time() - st
            self.num_checks_nn += 1

            print(s_next)
            
            collision_probability = 1.0 if (self.OBS and self.obstacle_check(s_next)) else 0.0
            
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
            stb = rospy.get_time()
            SA = np.concatenate((S, np.tile(a, (S.shape[0],1))), axis=1)
            S_next = []
            for sa in SA:
                st = rospy.get_time()
                sa_next = self.predict(sa)
                self.time_nn += rospy.get_time() - st
                self.num_checks_nn += 1
                S_next.append(sa_next)
            S_next = np.array(S_next)
            self.time_bnn += rospy.get_time() - stb
            self.num_checks_bnn += 1

            if self.OBS:
                # print "Checking obstacles..."
                failed_inx = []
                good_inx = []
                for i in range(S_next.shape[0]):
                    if self.obstacle_check(S_next[i,:]):
                        failed_inx.append(i)
                collision_probability = float(len(failed_inx))/float(S.shape[0])
                # node_probability = min(node_probability, node_probability2)

                if len(failed_inx):
                    good_inx = np.delete( np.array(range(S_next.shape[0])), failed_inx )
                    if len(good_inx) == 0: # All particles failed
                        S_next = []
                        mean = [0,0]
                        return {'next_states': S_next, 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': np.array([0.,0.]), 'collision_probability': 1.0}

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
            else:
                collision_probability = 0.0

            # print('svm time: ' + str(self.time_svm/self.num_checks_svm) + ', prediction time: ' + str(self.time_nn/self.num_checks_nn) + ', batch prediction time: ' + str(self.time_bnn/self.num_checks_bnn))

            mean = np.mean(S_next, 0) #self.get_mean_shift(S_next)
            return {'next_states': S_next.reshape((-1,)), 'mean_shift': mean, 'node_probability': node_probability, 'bad_action': bad_action, 'collision_probability': collision_probability}

    def obstacle_check(self, s):
        Obs = np.array([[-38, 117.1, 4.],
            [-33., 105., 4.],
            [-52.5, 105.2, 4.],
            [43., 111.5, 6.],
            [59., 80., 3.],
            [36.5, 94., 4.]
        ])
        f = 1.2 # inflate

        for obs in Obs:
            if np.linalg.norm(s[:2]-obs[:2]) <= f * obs[2]:
                return True
        return False

    # Predicts the next step by calling the GP class - repeats the same action 'n' times
    def GetTransitionRepeat(self, req):

        n = req.num_repeat

        TranReq = batch_transition()
        TranReq.states = req.states
        TranReq.action = req.action

        for _ in range(n):
            res = self.GetTransition(TranReq)
            TranReq.states = res['next_states']
            prob = res['node_probability']
            if prob < req.probability_threshold:
                break
        
        return {'next_states': res['next_states'], 'mean_shift': res['mean_shift'], 'node_probability': res['node_probability'], 'bad_action': res['bad_action'], 'collision_probability': res['collision_probability']}

    # Predicts the next step by calling the GP class
    def GetTransitionOneParticle(self, req):

        s = np.array(req.state)
        a = np.array(req.action)

        # Check which particles failed
        p = self.probability(s, a)
        node_probability = 1.0 - p

        # Propagate
        sa = np.concatenate((s, a), axis=0)
        s_next = self.predict(sa)   

        return {'next_state': s_next, 'node_probability': node_probability}

if __name__ == '__main__':
    try:
        NN = Spin_nn()
    except rospy.ROSInterruptException:
        pass