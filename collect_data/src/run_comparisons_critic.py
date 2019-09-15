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

# Set 10, 14nn , 0c_nn, 1c_nn- Don't delete!!!
# goals = [
# "-37, 119,  16,  16", 
# "-33, 102,  16,  16",
# "-40, 100,  16,  16",
# "-80, 80,  16,  16",
# "-50, 90,  16,  16",
# "50, 90,  16,  16",
# "40, 100,  16,  16",
# "-52, 112,  16,  16",
# "67, 80,  16,  16",
# "-63, 91,  16,  16",
# "75, 75, 16, 16",
# ]

# 6c_nn
# goals = [
# "-40, 82,  16,  16", 
# ]

# 7c_nn, 8c_nn
# goals = [
# # "-82, 60,  16,  16", 
# # "-52, 80,  16,  16", 
# # "52, 83,  16,  16", 
# "36, 95,  16,  16", 
# "0.5, 133,  16,  16", 
# "1, 109,  16,  16", 
# "-73, 60,  16,  16", 
# "85, 72,  16,  16", 
# "25, 100,  16,  16", 
# ]

# 10c_7
goals = [
"90, 60,  16,  16", 
"-90, 60,  16,  16", 
]

# 16c_7
goals = [
"49, 95,  16,  16", 
"62, 83,  16,  16", 
"41, 116,  16,  16", 
"70, 72,  16,  16", 
"20, 123,  16,  16", 
"16, 107,  16,  16", 
]

goals = [
# "-66, 80,  16,  16",
# "-41, 100,  16,  16", 
# "-62, 96,  16,  16", 
# "-49, 86,  16,  16", 
# "-55, 92,  16,  16",
# "59, 78,  16,  16",
# "31, 102,  16,  16",
# "60, 100,  16,  16",
# "52, 95,  16,  16",
# "-78, 67,  16,  16",
# "31, 125,  16,  16",
# "-26, 125,  16,  16",
# "0, 107,  16,  16",
# "3, 130,  16,  16",
# "-48, 114,  16,  16",
# "69, 78,  16,  16",
"78, 60, 16, 16",
]

# CRITIC NO SEQ
# nodes =[
# "naive_withCriticThreshold",
# "naive_withCriticCost",
# "naive_withCriticPredict",
# "naive",
# ]

# CRITIC WITH k=40 SEQ
nodes =[
#"naive_withCriticThreshold_withCriticSeq",
"naive_withCriticCost_withCriticSeq",
"naive",
# "naive_withCriticPredict_withCriticSeq",
]


SET_FOLDER = "set19c_7"
# ROLLOUT_FOLDER = "/home/akimmel/repositories/pracsys/src/"
# ROLLOUT_FOLDER = "/home/pracsys/catkin_ws/src/"
ROLLOUT_FOLDER = "/home/juntao/catkin_ws/src/"

NUM_RUNS = 1

GOAL_RADIUS = 5.5
TOTAL_PARTICLES = 100
# PROBABILITY_CONSTRAINT = 0.7
PROBABILITY_CONSTRAINT = 0.65
# PROBABILITY_CONSTRAINT = 0.001
NO_COLLISION_CONSTRAINT = 0.94
# SUCCESS_PROB_CONSTRAINT = 0.7
SUCCESS_PROB_CONSTRAINT = 0.45
FAILURE_CONSTANT = 100.0
CRITIC_THRESHOLD = 0.50
CRITIC_COST_MULTIPLIER = 10.0
# ]

#generate random goals
# NUM_RANDOM_GOALS = 200
# goals = []
# f=open("random_goals.txt", "a+")
# for x in range(NUM_RANDOM_GOALS): 
#     rand_x = random.uniform(-90,90)
#     rand_y = random.uniform(36,136)
#     rand_goal = str(rand_x) + "," + str(rand_y) + ",16,16"
#     goals.append(rand_goal)
#     f.write(rand_goal + "\n")

# C = np.loadtxt('/home/pracsys/catkin_ws/src/beliefspaceplanning/rollout_node/set/set8c_nn/random_goals.txt', delimiter=',', dtype=float)[:,:2]
# I = [1, 6, 31, 38, 91, 99, 123, 124, 134, 138]
# I = [38, 91, 99]#, 123, 124, 134, 138]


if __name__ == "__main__":
    for x in range(0, NUM_RUNS):
        count = 16
        for g in goals:

        # for ii in I:
        #     g = str(C[ii,0]) + "," + str(C[ii,1]) + ",16,16"
        #     count = ii

            for n in nodes:
                set_folder= "set_folder:=" + SET_FOLDER
                rollout_folder ="rollout_folder:=" + ROLLOUT_FOLDER
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
                critic_threshold="critic_threshold:=0.0"
                critic_cost_multiplier="critic_cost_multiplier:=1.0"
                use_critic_predict="use_critic_predict:=false"
                use_critic_cost="use_critic_cost:=false"
                use_critic_seq="use_critic_seq:=false"
                if "_withCriticThreshold" in n:
                    critic_threshold="critic_threshold:=" + str(CRITIC_THRESHOLD)
                if "_withCriticCost" in n:
                    use_critic_cost="use_critic_cost:=true"
                    critic_cost_multiplier="critic_cost_multiplier:=" + str(CRITIC_COST_MULTIPLIER)
                if "_withCriticPredict" in n:
                    use_critic_predict="use_critic_predict:=true"
                if "_withCriticSeq" in n:
                    use_critic_seq="use_critic_seq:=true"
                # experiment_filename="experiment_filename:=experiment"+str(count)+".txt"
                experiment_filename="experiment_filename:="+n+".txt"
                print node_name, goal_state, probability_constraint,no_collision_constraint, success_constraint
                goal_radius="goal_radius:=" + str(GOAL_RADIUS)
                subprocess.call(["roslaunch", "robust_planning", "run_comparisons_template.launch", set_folder, rollout_folder, node_name, critic_cost_multiplier, critic_threshold, use_critic_seq, use_critic_predict, use_critic_cost, goal_state, total_particles, probability_constraint, prune_probability, prune_covariance, goal_radius, experiment_filename, mean_only, use_svm_prediction, failure_constant, success_constraint, no_collision_constraint, random_seed])
            count = count + 1

