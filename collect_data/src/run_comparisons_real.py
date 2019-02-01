#!/usr/bin/env python

import sys
import subprocess

import rospy
import numpy as np
import time
import random
from std_msgs.msg import String, Float32MultiArray
from std_srvs.srv import Empty, EmptyResponse
from rollout_node.srv import rolloutReq, observation, IsDropped, TargetAngles
from sklearn.neighbors import NearestNeighbors 
from transition_experience import *
from collect_data.srv import sparse_goal

# goals = [
# "-26.03471520,  105.2378,  16,  16,  0.026,  0",
# "47,  111.2378,  16,  16,  0.026,  0",
# "60,  83.2378,  16,  16,  0.026,  0",
# "-72,  77.2378,  16,  16,  0.026,  0",
# "-69,  78.2378,  16,  16,  0.026,  0",
# "65,  80.2378,  16,  16,  0.026,  0",
# "50,  80.2378,  16,  16,  0.026,  0",
# "38.726146, 126.9431,  16,  16,  0.026,  0",
# "-5.043, 106.210,  16,  16,  0.026,  0",
# "-74.9059, 97.05,  16,  16,  0.026,  0",
# ]

# goals = [
# "-5.043, 106.210,  16,  16,  0.026,  0",
# "-74.9059, 97.05,  16,  16,  0.026,  0",
# "-72,77,  16,  16,  0.026,  0",
# "65,83,  16,  16,  0.026,  0",
# "-46,77,  16,  16,  0.026,  0",
# "40,100,  16,  16,  0.026,  0",
# "-26,105,  16,  16,  0.026,  0",
# "20,103,  16,  16,  0.026,  0"
# ]


# nodes =[
# "robust_particles",
# "robust_particles_pp",
# "robust_particles_pc",
# "robust_particles_pp_pc",
# "mean_only_particles",
# "naive_with_svm",
# "naive"
# ]


# goals = [
# "-77, 64,  16,  16,  0.026,  0",
# "80, 56,  16,  16,  0.026,  0",
# "-40, 82,  16,  16,  0.026,  0",
# "40, 82,  16,  16,  0.026,  0",
# "0, 106,  16,  16,  0.026,  0",
# "-16, 100,  16,  16,  0.026,  0",
# "16, 100,  16,  16,  0.026,  0",
# ]


# nodes =[
# "robust_particles_pc",
# "robust_particles_pp_pc",
# "mean_only_particles",
# "naive_with_svm",
# ]

# nodes =[
# "robust_particles_pc",
# "robust_particles_pp_pc",
# "mean_only_particles"
# ]

nodes =[
# "robust_particles_pc_svmHeuristic",
"naive_with_svm",
# "mean_only_particles",
]
## REAL GOALS
goals = [
"-5.043, 116.210,  16,  16",
"-20.9059, 97.05,  16,  16",
"20,105,  16,  16",
"40,110,  16,  16",
"-26,105,  16,  16",
"20,103,  16,  16"
]

GOAL_RADIUS = 3 #10
TOTAL_PARTICLES = 200
# PROBABILITY_CONSTRAINT = 0.7
PROBABILITY_CONSTRAINT = 0.6 #0.8
FAILURE_CONSTANT = 100.0

if __name__ == "__main__":
    for x in range(20):
        count = 0 #9
        for g in goals:
            for n in nodes:
                node_name = "node:="+ n + "_goal" + str(count) + "_run" + str(x)
                goal_state = "goal_state:="+ g
                total_particles = "total_particles:="
                probability_constraint = "minimum_prob:="
                mean_only="mean_only:="
                if "robust_particles" in n:
                    probability_constraint += str(PROBABILITY_CONSTRAINT)
                    mean_only+="false"
                    total_particles += str(TOTAL_PARTICLES)
                elif "naive_"in n:
                    probability_constraint += "0.5"
                    total_particles += str(1)
                    mean_only+="false"
                elif "mean_only_particles"in n:
                    probability_constraint += str(PROBABILITY_CONSTRAINT)
                    total_particles += str(TOTAL_PARTICLES)
                    mean_only+="true"
                else:
                    probability_constraint += str(PROBABILITY_CONSTRAINT)
                    total_particles += str(1)
                prune_covariance="prune_covariance:=false"
                if "_pc" in n:
                    prune_covariance= "prune_covariance:=true"
                prune_probability="prune_probability:=false"
                if "_pp" in n:
                    prune_probability= "prune_probability:=true"
                use_svm_prediction="use_svm_prediction:=false"
                failure_constant="failure_constant:=0"
                if "_svmHeuristic" in n:
                    use_svm_prediction="use_svm_prediction:=true"
                    failure_constant="failure_constant:="+str(FAILURE_CONSTANT)
                experiment_filename="experiment_filename:=experiment"+str(count)+".txt"
                print node_name, goal_state, probability_constraint
                goal_radius="goal_radius:=" + str(GOAL_RADIUS)
                subprocess.call(["roslaunch", "robust_planning", "run_comparisons_real_template.launch", node_name, goal_state, total_particles, probability_constraint, prune_probability, prune_covariance, goal_radius, experiment_filename, mean_only, use_svm_prediction, failure_constant])
            count = count + 1

