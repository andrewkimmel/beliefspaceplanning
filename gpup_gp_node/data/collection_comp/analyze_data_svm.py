#!/usr/bin/env python

# from scipy.signal import medfilt

import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.preprocessing import StandardScaler

def medfilter(x, W):
    w = int(W/2)
    x_new = np.copy(x)
    for i in range(0, x.shape[0]):
        if i < w:
            x_new[i] = np.mean(x[:i+w])
        elif i > x.shape[0]-w:
            x_new[i] = np.mean(x[i-w:])
        else:
            x_new[i] = np.mean(x[i-w:i+w])
    return x_new

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/collection_comp/'

if 0:
    M = {}

    file = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_raw_discrete_v13.obj' 
    with open(file, 'rb') as filehandler:
        M['p'] = pickle.load(filehandler)

    # Get test data
    Set = 'p'
    K = M[Set]
    states = np.array([item[0][:4] for item in K])
    actions = np.array([item[1] for item in K])
    doner = np.array([item[3] for item in K])

    for i in range(doner.shape[0]):
        if doner[i]:
            doner[i-2:i] = True

    states = states[:, [0, 1, 2, 3]]
    states_actions = np.concatenate((states, actions), axis=1)
    
    ni = 50
    T = np.where(doner)[0]
    inx_fail = T[np.random.choice(T.shape[0], ni, replace=False)]
    T = np.where(np.logical_not(doner))[0]
    inx_suc = T[np.random.choice(T.shape[0], ni, replace=False)]
    SA_test = np.concatenate((states_actions[inx_fail], states_actions[inx_suc]), axis=0)
    done_test = np.concatenate((doner[inx_fail], doner[inx_suc]), axis=0)

    ix = np.concatenate((inx_fail, inx_suc), axis=0)
    for i in sorted(ix, reverse=True):
        del M['p'][i]

    file = path + 'sim_raw_discrete_v100_random10A.obj' 
    with open(file, 'rb') as filehandler:
        M['10'] = pickle.load(filehandler)

    file = path + 'sim_raw_discrete_v100_random50A.obj' 
    with open(file, 'rb') as filehandler:
        M['50'] = pickle.load(filehandler)

    n = np.min([len(M[Set]) for Set in M.keys()])

    L = np.linspace(7000, n, num = 10000)

    R = []
    for l in L:
        r = {}
        for Set in M.keys():
            K = M[Set][0:int(l)]
            states = np.array([item[0][:4] for item in K])
            actions = np.array([item[1] for item in K])
            doner = np.array([item[3] for item in K])

            for i in range(doner.shape[0]):
                if doner[i]:
                    doner[i-2:i] = True

            states = states[:, [0, 1, 2, 3]]
            states_actions = np.concatenate((states, actions), axis=1)

            inx_fail = np.where(doner)[0]
            T = np.where(np.logical_not(doner))[0]
            print "Number of failed states " + str(inx_fail.shape[0])
            inx_suc = T[np.random.choice(T.shape[0], inx_fail.shape[0], replace=False)]
            SA = np.concatenate((states_actions[inx_fail], states_actions[inx_suc]), axis=0)
            done = np.concatenate((doner[inx_fail], doner[inx_suc]), axis=0)

            print 'Fitting SVM...'
            clf = svm.SVC( probability=True, class_weight='balanced', C=1.0 )
            clf.fit( list(SA), 1*done )

            print "For set " + Set + ':'
            s = 0
            s_suc = 0; c_suc = 0
            s_fail = 0; c_fail = 0
            for i in range(SA_test.shape[0]):
                p = clf.predict_proba(SA_test[i].reshape(1,-1))[0]
                fail = p[1]>0.5
                # print p, done_test[i], fail
                s += 1 if fail == done_test[i] else 0
                if done_test[i]:
                    c_fail += 1
                    s_fail += 1 if fail else 0
                else:
                    c_suc += 1
                    s_suc += 1 if not fail else 0
            print 'Success rate: ' + str(float(s)/SA_test.shape[0]*100)
            r[Set] = float(s)/SA_test.shape[0]
        R.append(r)

    with open(path + 'svm_analysis.obj', 'wb') as f:
        pickle.dump([L, R], f) 
else:
    with open(path + 'svm_analysis.obj', 'rb') as f:
        L, R = pickle.load(f) 

# R = np.array(R)
E = []
for r in R:
    E.append(r.values())
E = np.array(E)
S = R[0].keys()

plt.figure(1)
for i in range(len(S)):
    if i == 1:
        continue
    E[:,i] = medfilter(E[:,i], 101)
    plt.plot(L, E[:,i]*100, label=S[i])
plt.legend()
plt.title('SVM accuracy improvement')
plt.xlabel('Size of data')
plt.ylabel('Success rate [%]')
plt.savefig(path + 'svm', dpi=300)
plt.show()