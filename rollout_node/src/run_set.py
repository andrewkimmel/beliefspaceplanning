#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from rollout_node.srv import rolloutReq
import time
import glob

rollout_srv = rospy.ServiceProxy('/rollout/rollout', rolloutReq)

rospy.init_node('run_rollout_set', anonymous=True)
rate = rospy.Rate(15) # 15hz
state_dim = 6

set_mode = 'mean_only'

path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'

############################# Rollout ################################
if 0:

    files = glob.glob(path + "*.txt")

    for i in range(len(files)):

        action_file = files[i]
        pklfile = action_file[:-3] + 'pkl'

        print('Rolling-out file number ' + str(i+1) + ': ' + action_file + '.')

        A = np.loadtxt(action_file, delimiter=',', dtype=float)[:,:2]

        Af = A.reshape((-1,))
        Pro = []
        for j in range(100):
            print("Rollout number " + str(j) + ".")
            
            Sro = np.array(rollout_srv(Af).states).reshape(-1,state_dim)

            Pro.append(Sro)

            if not (j % 5):
                with open(pklfile, 'w') as f: 
                    pickle.dump(Pro, f)

############################# Plot ################################

if 1:

    files = glob.glob(path + "*.pkl")

    for k in range(len(files)):

        pklfile = files[k]

        for j in range(len(pklfile)-1, 0, -1):
            if pklfile[j] == '/':
                break
        file_name = pklfile[j+1:-4]

        print('Plotting file number ' + str(k+1) + ': ' + file_name)
        
        with open(pklfile) as f:  
            Pro = pickle.load(f) 

        maxR = np.max([x.shape[0] for x in Pro])
        c = np.sum([(1 if x.shape[0]==maxR else 0) for x in Pro])

        Smean = []
        Sstd = []
        for i in range(maxR):
            F = []
            for j in range(len(Pro)): 
                if Pro[j].shape[0] > i:
                    F.append(Pro[j][i])
            Smean.append( np.mean(np.array(F), axis=0) )
            Sstd.append( np.std(np.array(F), axis=0) )
        Smean = np.array(Smean)
        Sstd = np.array(Sstd)

        print("Roll-out success rate: " + str(float(c) / len(Pro)*100) + "%")

        fig = plt.figure(k)
        for S in Pro:
            plt.plot(S[:,0], S[:,1], 'r')

        plt.plot(Smean[:,0], Smean[:,1], '-b', label='rollout mean')
        X = np.concatenate((Smean[:,0]+Sstd[:,0], np.flip(Smean[:,0]-Sstd[:,0])), axis=0)
        Y = np.concatenate((Smean[:,1]+Sstd[:,1], np.flip(Smean[:,1]-Sstd[:,1])), axis=0)
        plt.fill( X, Y , alpha = 0.5 , color = 'b')
        plt.plot(Smean[:,0]+Sstd[:,0], Smean[:,1]+Sstd[:,1], '--b', label='rollout mean')
        plt.plot(Smean[:,0]-Sstd[:,0], Smean[:,1]-Sstd[:,1], '--b', label='rollout mean')       
        plt.title(file_name + ", suc. rate: " + str(float(c) / len(Pro)*100) + "%")
        
    plt.show()