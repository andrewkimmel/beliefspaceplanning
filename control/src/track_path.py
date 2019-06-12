#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import pickle
from control.srv import pathTrackReq
import time
import glob
from scipy.io import loadmat

rollout = 1

# path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'
# path = '/home/juntao/catkin_ws/src/beliefspaceplanning/rollout_node/set/' + set_mode + '/'

# comp = 'juntao'
comp = 'pracsys'

# Line shape
x = np.linspace(0, 20, 30)
y = np.zeros(x.shape)
line = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)

# Rectangle shape
x = np.concatenate((np.linspace(0, 8, 16), np.linspace(8, 8, 10), np.linspace(8, -8, 32), np.linspace(-8, -8, 10), np.linspace(-8, 0, 16)))
y = np.concatenate((np.linspace(0, 0, 16), np.linspace(0, -3, 10), np.linspace(-3, -3, 32), np.linspace(-3, 0, 10), np.linspace(0, 0, 16)))
rec = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)

# Rectangle2 shape
x = np.concatenate((np.linspace(0, 8, 16), np.linspace(8, 4, 20), np.linspace(4, -4, 32), np.linspace(-4, -8, 20), np.linspace(-8, 0, 16)))
y = np.concatenate((np.linspace(0, 0, 16), np.linspace(0, -7, 20), np.linspace(-7, -7, 32), np.linspace(-7, 0, 20), np.linspace(0, 0, 16)))
rec2 = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)

# Circle shape
th = np.linspace(np.pi/2,np.pi/2 + 2*np.pi,100)
r = 3
x = r * np.cos(th)
y = 0.5 * r * np.sin(th) - 0.5*r
circle = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)

shapes = {'line': line, 'rectangle': rec2, 'circle': circle}
# shapes = {'rectangle': rec2}
# shapes = {'circle': circle}



############################# Rollout ################################
if rollout:
    track_srv = rospy.ServiceProxy('/control', pathTrackReq)
    rospy.init_node('run_control_track', anonymous=True)
    state_dim = 4

    path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/control/paths/'
    results_path = '/home/' + comp + '/catkin_ws/src/beliefspaceplanning/control/paths/'

    for shape in shapes:
        print('Rolling-out ' + shape + '...')

        S = shapes[shape]

        Pro = []
        Aro = []
        Suc = []
        pklfile = path + shape + '.pkl'
        for j in range(1):
            res = track_srv(S.reshape((-1,)))
            Sreal = np.array(res.real_path).reshape(-1, 8)
            Areal = np.array(res.actions).reshape(-1, 2)
            success = res.success

            Pro.append(Sreal)
            Aro.append(Areal)
            Suc.append(success)

            with open(pklfile, 'w') as f: 
                pickle.dump([Pro, Aro, Suc], f)

        Straj = S + Pro[0][0,:2]
        plt.plot(Straj[:,0], Straj[:,1], '--k')

        for S, suc in zip(Pro, Suc):
            if suc:
                plt.plot(S[:,0], S[:,1], '-b')
            else:
                plt.plot(S[:,0], S[:,1], '-r')

        # plt.title(file_name + ", suc. rate: " + str(c) + "%, " + "goal suc.: " + str(p) + "%")
        plt.axis('equal')
        plt.savefig(results_path + shape + '.png', dpi=250)
        # plt.show()

############################# Plot ################################
