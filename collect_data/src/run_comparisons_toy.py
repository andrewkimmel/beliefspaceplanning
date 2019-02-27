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
## ROBUST PLUS GOALS part 4 - Avishai changed this

nodes =[
"robust_particles_pc_svmHeuristic",
"naive_with_svm",
"mean_only_particles",
]

goals = [
"0.9, -0.9",
# "0.9,0.9",
# "0.9,0"
]

# starts = [
# "-.9, -.9",
# # "0,0",
# # "-.95, .95"
# ]

s= "-.9,-.9"

GOAL_RADIUS = .95
TOTAL_PARTICLES = 420
# PROBABILITY_CONSTRAINT = 0.7
PROBABILITY_CONSTRAINT = 0.85
# PROBABILITY_CONSTRAINT = 0.6
SUCCESS_PROBABILITY_CONSTRAINT = 0.5
FAILURE_CONSTANT = 10.0

if __name__ == "__main__":
    for x in range(20):
        count = 0
        for g in goals:
            for n in nodes:
                random_seed = "random_seed:=" + str(seed[x])
                node_name = "node:="+ n + "_goal" + str(count) + "_run" + str(x)
                goal_state = "goal_state:="+ g
                start_state = "start_state:="+ s
                total_particles = "total_particles:="
                probability_constraint = "minimum_drop_prob:="
                success_constraint = "minimum_success_prob:=" + str(SUCCESS_PROBABILITY_CONSTRAINT)
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
                print node_name, goal_state, probability_constraint
                goal_radius="goal_radius:=" + str(GOAL_RADIUS)
                subprocess.call(["roslaunch", "robust_planning", "run_comparisons_toy_template.launch", node_name, start_state, goal_state, total_particles, probability_constraint, success_constraint, prune_probability, prune_covariance, goal_radius, experiment_filename, mean_only, use_svm_prediction, failure_constant, random_seed])
            count = count + 1
