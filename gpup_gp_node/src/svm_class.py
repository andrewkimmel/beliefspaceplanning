#!/usr/bin/env python

import numpy as np
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import var

class svm_failure():

    r = 0.1
    OBS = True

    def __init__(self, discrete = True):

        self.mode = 'discrete' if discrete else 'cont'

        self.load_data()

        print 'All set!'

    def load_data(self):

        # path = '/home/akimmel/repositories/pracsys/src/beliefspaceplanning/gpup_gp_node/data/'
        path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
        File = 'svm_data_' + self.mode + '_v' + str(var.data_version_) + '_d' + str(var.dim_) + '_m' + str(var.stepSize_) + '.obj' # <= File name here!!!!!

        print('Loading data from ' + File)
        with open(path + File, 'rb') as f: 
            self.SA, self.done = pickle.load(f)
        print('Loaded svm data.')      

        # Normalize
        scaler = StandardScaler()
        self.SA = scaler.fit_transform(self.SA)
        self.x_mean = scaler.mean_
        self.x_std = scaler.scale_

        print 'Fitting SVM...'
        self.clf = svm.SVC( probability=True, class_weight='balanced', C=1.0 )
        self.clf.fit( list(self.SA), 1*self.done )
        print 'SVM ready with %d classes: '%len(self.clf.classes_) + str(self.clf.classes_)

    def probability(self, s, a):
        if self.OBS and self.obstacle_check(s):
            return 1., True

        sa = np.concatenate((s,a), axis=0).reshape(1,-1)

        # Normalize
        sa = (sa - self.x_mean) / self.x_std

        p = self.clf.predict_proba(sa)[0][1]

        return p, self.clf.predict(sa)

    def obstacle_check(self, s):
        # Obs1 = np.array([42, 90, 12.])
        # Obs2 = np.array([-45, 101, 7.])
        # f = 1.15 # inflate
        Obs1 = np.array([33, 110, 4.]) # Right
        Obs2 = np.array([-27, 118, 2.5]) # Left
        f = 1.5 # inflate

        if np.linalg.norm(s[:2]-Obs1[:2]) < f * Obs1[2] or np.linalg.norm(s[:2]-Obs2[:2]) < f * Obs2[2]:
            return True
        else:
            return False


