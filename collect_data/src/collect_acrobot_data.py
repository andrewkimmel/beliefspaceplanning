#!/usr/bin/env python

import sys
import subprocess

import rospy
import numpy as np
import time
import random

for x in range(20):
    subprocess.call(["roslaunch", "prx_input", "ao_rrt_acrobot.launch"])
