#!/usr/bin/env python

import numpy as np 
import pickle

def combine(file1, file2, f):

    print 'Loading ' + file1 + '...'
    with open(file1, 'rb') as filehandler:
        M1 = pickle.load(filehandler)
    print('Loaded %d points.'%len(M1))

    print 'Loading ' + file2 + '...'
    with open(file2, 'rb') as filehandler:
        M2 = pickle.load(filehandler)
    print('Loaded %d points.'%len(M2))

    M = M1 + M2

    print len(M1), len(M2), len(M), len(M1)+len(M2)

    print('Saving data with %d points...'%len(M))
    file_pi = open(f, 'wb')
    pickle.dump(M, file_pi)
    print('Saved transition data of size %d.'%len(M))
    file_pi.close()

f1 = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_raw_discrete_v12a_bu.obj'
f2 = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_raw_discrete_v12b_bu.obj'
f = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_raw_discrete_v12.obj'

combine(f1, f2, f)

