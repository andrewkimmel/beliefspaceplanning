#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import glob


set_modes = ['robust_particles_pc', 'naive_with_svm', 'mean_only_particles']
path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/set19_nn/results/'

W = 1200
H = 800

for i in range(3):
    P = []
    for set_mode in set_modes:
        files = glob.glob(path + set_mode + "*.png")
        F = '-1'
        for File in files:
            if int(File[File.find('goal')+4]) == i:
                F = File
                break
        if F == '-1':
            I = np.zeros((H,W,4))
        else:
            I = plt.imread(F)

            if i == 0:
                x = 1200
                y = 760
            elif i == 1:
                x = 1200
                y = 760
            elif i == 2:
                x = 1254
                y = 1782
                # plt.imshow(I)
                # plt.show()
                # exit(1)
            else:
                continue
            I = I[x:x+H,y:y+W,:]

        P.append(I)

    I = np.concatenate((P[0], P[1], P[2]), axis=1)

    # plt.imshow(I)
    plt.imsave(path + 'robust_naive_mean_' + str(i) + '.png', I)