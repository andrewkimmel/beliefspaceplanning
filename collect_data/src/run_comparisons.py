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
"robust_particles_pc", 
# "naive_with_svm",
# "mean_only_particles"
]
## ROBUST PLUS GOALS part 1
# goals = [
# "-5.043, 106.210,  16,  16,  0.026,  0",
# "-74.9059, 97.05,  16,  16,  0.026,  0",
# "65,83,  16,  16,  0.026,  0",
# "40,100,  16,  16,  0.026,  0",
# "-26,105,  16,  16,  0.026,  0",
# "20,103,  16,  16,  0.026,  0"
# ]

## ROBUST PLUS GOALS part 3- Avishai changed this

# goals = [
# "-40, 100.210,  16,  16,  0.026,  0",
# "-60.9059, 90.05,  16,  16,  0.026,  0",
# "0,110,  16,  16,  0.026,  0",
# "77,90,  16,  16,  0.026,  0",
# "42,85,  16,  16,  0.026,  0",
# "20,127,  16,  16,  0.026,  0"
# ]

seed = [
121352,
517517,
90219,
829329,
854750,
315604,
450032,
299577,
460861,
102557,
502620,
90152,
262304,
455762,
986361,
850173,
946174,
36375,
908293,
438016,
943726,
641619,
575966,
799277,
968928,
299693,
528931,
941909,
524368,
223754,
59924,
72583,
530268,
686644,
978683,
671628,
965098,
293854,
536775,
727240,
286633,
756622,
810768,
776987,
857337,
421650,
99062,
117072,
896704,
132096,
131234,
98964]


## Set 1 - Don't delete!!!
# goals = [
# "17, 117,  16,  16",
# "75, 75,  16,  16",
# "-83, 66,  16,  16",
# "-52, 77,  16,  16",
# "48, 114,  16,  16",
# "-31, 100,  16,  16",
# "5.4, 108,  16,  16",
# "87, 65,  16,  16",
# ]

## Set 1 & 2 - Don't delete!!!
# goals = [
# "-24, 115,  16,  16",
# " 58, 76,  16,  16",
# "-56, 90,  16,  16",
# " 79, 76,  16,  16",
# " -40, 121,  16,  16",
# "-66, 97,  16,  16",
# "-46, 77,  16,  16",
# "-73, 63,  16,  16",
# " 60, 100,  16,  16",
# " 35, 106,  16,  16",
# " 27, 104,  16,  16",
# " 4.5, 109,  16,  16",
# " -27, 108,  16,  16",
# " 57, 110,  16,  16",
# ]

## Set 3 - Don't delete!!!
# goals = [
# "40, 95,  16,  16",
# "50, 111,  16,  16",
# "25, 98,  16,  16",
# "-32, 104,  16,  16",
# ]

# # Set 4 - Don't delete!!!
# goals = [
# "-37, 119,  16,  16, 0, 0, 0, 0",
# "-33, 102,  16,  16, 0, 0, 0, 0",
# "-22, 129,  16,  16, 0, 0, 0, 0",
# "-52, 112,  16,  16, 0, 0, 0, 0",
# "67, 80,  16,  16, 0, 0, 0, 0",
# "-63, 91,  16,  16, 0, 0, 0, 0",
# ]

## Set 5 - Don't delete!!!
# goals = [
# "50, 111,  16,  16",
# ]

## Set 8 & 9 - Don't delete!!!
# goals = [
# "-37, 119,  16,  16, 0, 0, 0, 0",
# "-33, 102,  16,  16, 0, 0, 0, 0",
# "-60, 90,  16,  16, 0, 0, 0, 0",
# "-40, 100,  16,  16, 0, 0, 0, 0",
# "-80, 65,  16,  16, 0, 0, 0, 0",
# "-80, 80,  16,  16, 0, 0, 0, 0",
# "-50, 90,  16,  16, 0, 0, 0, 0",
# "60, 90,  16,  16, 0, 0, 0, 0",
# "80, 80,  16,  16, 0, 0, 0, 0",
# "50, 90,  16,  16, 0, 0, 0, 0",
# "40, 100,  16,  16, 0, 0, 0, 0",
# "80, 65,  16,  16, 0, 0, 0, 0",
# "-52, 112,  16,  16, 0, 0, 0, 0",
# "67, 80,  16,  16, 0, 0, 0, 0",
# "-63, 91,  16,  16, 0, 0, 0, 0",
# ]

# Set 10, 14nn - Don't delete!!!
goals = [
"-37, 119,  16,  16", 
"-33, 102,  16,  16",
"-40, 100,  16,  16",
"-80, 80,  16,  16",
"-50, 90,  16,  16",
"50, 90,  16,  16",
"40, 100,  16,  16",
"-52, 112,  16,  16",
"67, 80,  16,  16",
"-63, 91,  16,  16",
]


## Set 15 - Don't delete!!!
# goals = [
# "-40, 97,  16,  16",
# ]

## Set 18, 19 - Don't delete!!!
# goals = [
# "-59, 90,  16,  16",
# # "-50, 90,  16,  16",
# ]

## Set 21 - Don't delete!!!
goals = [
# "-58, 80,  16,  16",
# "50,78,  16,  16",
# "73,76,  16,  16",
"-26,96,  16,  16",
# "57,103,  16,  16",
]

NUM_RUNS = 1

GOAL_RADIUS = 7
TOTAL_PARTICLES = 100
# PROBABILITY_CONSTRAINT = 0.7
PROBABILITY_CONSTRAINT = 0.65
NO_COLLISION_CONSTRAINT = 0.94
# SUCCESS_PROB_CONSTRAINT = 0.7
SUCCESS_PROB_CONSTRAINT = 0.45
FAILURE_CONSTANT = 100.0
# ]

if __name__ == "__main__":
    for x in range(NUM_RUNS):
        count = 0
        for g in goals:
            for n in nodes:
                random_seed = "random_seed:=" + str(seed[x])
                node_name = "node:="+ n + "_goal" + str(count) + "_run" + str(x)
                goal_state = "goal_state:="+ g
                total_particles = "total_particles:="
                no_collision_constraint = "minimum_no_collision:=" + str(NO_COLLISION_CONSTRAINT)
                success_constraint ="minimum_success_prob:=" + str(SUCCESS_PROB_CONSTRAINT)
                probability_constraint = "minimum_prob:="
                mean_only="mean_only:="
                if "robust_particles" in n:
                    probability_constraint += str(PROBABILITY_CONSTRAINT)
                    mean_only+="false"
                    total_particles += str(TOTAL_PARTICLES)
                elif "naive_with_svm"in n:
                    probability_constraint += str(PROBABILITY_CONSTRAINT)
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
                # experiment_filename="experiment_filename:=experiment"+str(count)+".txt"
                experiment_filename="experiment_filename:="+n+".txt"
                print node_name, goal_state, probability_constraint,no_collision_constraint, success_constraint
                goal_radius="goal_radius:=" + str(GOAL_RADIUS)
                subprocess.call(["roslaunch", "robust_planning", "run_comparisons_template.launch", node_name, goal_state, total_particles, probability_constraint, prune_probability, prune_covariance, goal_radius, experiment_filename, mean_only, use_svm_prediction, failure_constant, success_constraint, no_collision_constraint, random_seed])
            count = count + 1

