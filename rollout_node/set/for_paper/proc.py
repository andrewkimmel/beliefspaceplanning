#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import glob


set_modes = ['robust_particles_pc', 'naive_with_svm', 'mean_only_particles']
paths = ['/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/set19_nn/results/',
       '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/set20_nn/results/']
des_path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/for_paper/'

W = 1200
H = 800

b = 0
for path in paths:
    for i in range(5):
        P = []
        hj = 0
        for set_mode in set_modes:
            files = glob.glob(path + set_mode + "*.png")
            F = '-1'
            for File in files:
                if int(File[File.find('goal')+4]) == i:
                    F = File
                    break
            if (path.find('set20_nn') > 0 or F == '-1') and not (i == 0 or i == 2 or i == 3):
                continue

            if F == '-1':
                I = np.zeros((H,W,4))
            else:
                I = plt.imread(F)
                hj += 1

                if b == 0:
                    x = 1200
                    y = 760
                elif b == 1:
                    x = 1200
                    y = 760
                elif b == 2 or b == 3:
                    x = 1254
                    y = 1782
                elif b == 4:
                    x = 1254
                    y = 1782
                elif b == 5:
                    x = 1200
                    y = 760
                else:
                    x = 1
                    y = 1
                    # continue
                I = I[x:x+H,y:y+W,:]
            
            plt.imsave(des_path + set_mode + '_goal' + str(b) + '.png', I)

            P.append(I)

        if hj == 0:
            continue
        I = np.concatenate((P[0], P[1], P[2]), axis=1)

        # plt.imshow(I)
        plt.imsave(des_path + 'robust_naive_mean_goal' + str(b) + '.png', I)
        b += 1