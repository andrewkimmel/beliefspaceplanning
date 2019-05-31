#!/usr/bin/env python

import numpy as np
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os.path

data_version_ = 14
dim_ = 4
stepSize_ = 1


class svm_failure():

    r = 0.1

    def __init__(self, discrete = True):

        self.mode = 'discrete' if discrete else 'cont'

        self.load_data()

        print('All set!')


    def load_data(self):

        # path = '/home/akimmel/repositories/pracsys/src/beliefspaceplanning/gpup_gp_node/data/'
        self.path = '/home/juntao/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'
        File = 'svm_data_' + self.mode + '_v' + str(data_version_) + '_d' + str(dim_) + '_m' + str(stepSize_) + '.obj' # <= File name here!!!!!

        self.postfix = '_v' + str(data_version_) + '_d' + str(dim_) + '_m' + str(stepSize_)
        if os.path.exists(self.path + 'svm_fit_' + self.mode + self.postfix + '.obj'):
            with open(self.path + 'svm_fit_' + self.mode + self.postfix + '.obj', 'rb') as f: 
                self.clf, self.x_mean, self.x_std = pickle.load(f, encoding='latin1') # Latin1 in python3
            print('[SVM] Loaded svm fit.')
        else:
            print('[SVM] Loading data from ' + File)
            with open(self.path + File, 'rb') as f: 
                self.SA, self.done = pickle.load(f, encoding='latin1')
            print('[SVM] Loaded svm data.')       

            # Normalize
            scaler = StandardScaler()
            self.SA = scaler.fit_transform(self.SA)
            self.x_mean = scaler.mean_
            self.x_std = scaler.scale_
        
            print('Fitting SVM...')
            self.clf = svm.SVC( probability=True, class_weight='balanced', C=1.0 )
            self.clf.fit( list(self.SA), 1*self.done )

            with open(self.path + 'svm_fit_' + self.mode +  self.postfix + '.obj', 'wb') as f: 
                pickle.dump([self.clf, self.x_mean, self.x_std], f)

        print('SVM ready with %d classes: '%len(self.clf.classes_) + str(self.clf.classes_))

    def probability(self, s, a):
        
        sa = np.concatenate((s,a), axis=0).reshape(1,-1)

        # Normalize
        sa = (sa - self.x_mean) / self.x_std

        p = self.clf.predict_proba(sa)[0][1]
        # pp = self.clf.predict(sa)

        return p#, pp


