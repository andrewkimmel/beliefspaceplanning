#!/usr/bin/env python

import numpy as np
import glob
# from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

files = glob.glob("*.png")
# 
for F in files:
    if F.find('cropped') > -1:
        continue
    print
    print F
    I=mpimg.imread(F)

    num = int(F[F.find('goal')+4])
    print "Goal " + str(num)
    print I.shape

    h = 600
    w = 800
    
    # plt.imshow(I)
    # plt.show()

    if num == 1:
        x, y = 470, 820
    elif num == 2:
        x, y = 730, 650
        h, w = 900, 1200
    elif num == 3:
        x, y = 1200, 800
    elif num == 4:
        x, y = 470, 778
    elif num == 5:
        x, y = 520, 778
    I_cropped = I[y:y+h, x:x+w, :]

    # plt.imshow(I_cropped)

    print I_cropped.shape, float(I_cropped.shape[1])/I_cropped.shape[0]

    # mpimg.imsave('cropped_' + F, I_cropped)
    # plt.show()
    # exit(1)