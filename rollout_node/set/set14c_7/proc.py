#!/usr/bin/env python

import numpy as np
import glob
# from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

files = glob.glob("*.png")
# 
for F in files:
    print F
    I=mpimg.imread(F)
    print I.shape
    I_cropped = I[810:1225, 1186:1923, :]
    # plt.imshow(I_cropped)

    mpimg.imsave('cropped_' + F, I_cropped)
    # plt.show()
    # exit(1)