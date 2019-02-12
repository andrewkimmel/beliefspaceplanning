#!/usr/bin/env python

import sys
import subprocess

import rospy
import numpy as np
import time
import random
from collect_data.msg import plan_for_goal_request

TOTAL_PARTICLES = 100
FAILURE_CONSTANT = 100.0

count = 0

def planCallback(msg):
    g = ""
    for st in msg.goal_state:
        g += str(st)
    print "Received goal state: " , g

    if (msg.goal_state.size() < 6):
        g+= "16,  16,  0.026,  0"

    print "---- corrected dimension goal state: " , g

    GOAL_RADIUS = msg.goal_radius
    PROBABILITY_CONSTRAINT = msg.probability_success_threshold
    TIME_LIMIT = msg.time_limit
    n = robust_particles_pc_svmHeuristic


    planning_time_limit="planning_time_limit:=" + str(TIME_LIMIT)
    node_name = "node:="+ n + "_goal" + str(count) + "_run" + str(x)
    goal_state = "goal_state:="+ g
    total_particles = "total_particles:="
    probability_constraint = "minimum_prob:="
    mean_only="mean_only:="
    if "robust_particles" in n:
        probability_constraint += str(PROBABILITY_CONSTRAINT)
        mean_only+="false"
        total_particles += str(TOTAL_PARTICLES)
    elif "naive_with_svm"in n:
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
    print node_name, goal_state, probability_constraint, planning_time_limit
    goal_radius="goal_radius:=" + str(GOAL_RADIUS)
    subprocess.call(["roslaunch", "robust_planning", "run_comparisons_template.launch", node_name, goal_state, total_particles, probability_constraint, prune_probability, prune_covariance, goal_radius, experiment_filename, mean_only, use_svm_prediction, failure_constant, planning_time_limit])
    count = count + 1


if __name__ == "__main__":

    rospy.init_node('planning_server', anonymous=True)
    rospy.Subscriber('/planning/request', plan_for_goal_request, planCallback)

    while not rospy.is_shutdown():
        rospy.spin()
