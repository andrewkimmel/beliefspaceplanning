#!/usr/bin/env python

import numpy as np
import time
import random
import pickle
import os.path
import matplotlib.pyplot as plt
from transition_experience import *


def main():
    texp = transition_experience(discrete=True)

    # texp.save_to_file()
    texp.process_transition_data(stepSize = 1, plot = False, mode = 1)
    # texp.reduce_data(mode = 3)
    texp.process_svm(stepSize = 1, mode = 1)

    texp.plot_data()

    return 1


if __name__=='__main__':
    main()
