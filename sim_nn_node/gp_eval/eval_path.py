#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gpup_gp_node.srv import one_transition
from sim_nn_node.srv import critic_seq

Set = '17c_7'
# goal = 0
# run = 1

# o_srv = rospy.ServiceProxy('/nn/transitionOneParticle', one_transition)
c_srv = rospy.ServiceProxy('/nn/critic_seq', critic_seq)
rospy.init_node('eval_path', anonymous=True)

path = '/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/set' + Set + '/'

v = dict()
for run in range(6):
    for goal in range(10):
        names = ['naive_withCriticCost_withCriticSeq_goal' + str(goal) + '_run' + str(run), 'naive_goal' + str(goal) + '_run' + str(run)]
        for name in names:
            trajfile = path + name + '_traj.txt'
            action_file = path + name + '_plan.txt'

            try:
                S = np.loadtxt(trajfile, delimiter=',', dtype=float)[:,:4]
                A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]
            except:
                continue

            print '-----------------------------'
            print name, run, goal

            i = 1
            l_prior = 40
            e = 0
            n = 0
            while np.all(A[i] == A[i-1]):
                i += 1
            while i < A.shape[0]:
                
                s = S[i]
                a = A[i]

                l = 1
                while i+l < A.shape[0] and np.all(A[i+l] == a):
                    l += 1

                # print i, l, i + ls

                H = A[np.maximum(0, i-l_prior):i-1, :]
                if H.shape[0] < l_prior:
                    H = np.concatenate((np.zeros((l_prior-H.shape[0], 2)), H), axis=0) 

                e += c_srv(s, a, l, H.reshape(-1,1)).err
                n += 1

                i += l
            
            if name.find('withCriticCost') > 0:
                key = 'critic' + str(goal)
            else:
                key = 'nocritic' + str(goal)
            try:
                v[key].append(e)
            except:
                v[key] = []    
                v[key].append(e)            

            print "Error: " + str(e / n), e

for k in v.keys():
    print str(k) + ':'
    print np.mean(np.array(v[k]))


