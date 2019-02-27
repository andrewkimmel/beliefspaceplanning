#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String, Float32MultiArray
from rollout_node.srv import rolloutReq, rolloutReqFile, plotReq, observation, IsDropped, TargetAngles
import numpy as np
import matplotlib.pyplot as plt
import pickle
from rollout_node.srv import gets

obj_pos = np.array([0., 0.])
gripper_load = np.array([0., 0.])
gripper_closed = "open"

def callbackGripperStatus(msg):
    global gripper_closed
    gripper_closed = msg.data == "closed"

def callbackObj(msg):
    global obj_pos
    Obj_pos = np.array(msg.data)
    obj_pos = Obj_pos[:2] * 1000

def callbackGripperLoad(msg):
    global gripper_load
    gripper_load = np.array(msg.data)

rospy.init_node('test_start', anonymous=True)
reset_srv = rospy.ServiceProxy('/hand_control/ResetGripper', Empty)
rospy.Subscriber('/hand_control/gripper_status', String, callbackGripperStatus)
rospy.Subscriber('/hand/obj_pos', Float32MultiArray, callbackObj)
rospy.Subscriber('/gripper/load', Float32MultiArray, callbackGripperLoad)


rate = rospy.Rate(10) 

if 0:
    # S = []
    # L = []
    with open('gazebo_start_states.obj', 'rb') as f:
        S, L = pickle.load(f)
    while len(S) < 1000:
        print "Run ", len(S)
        # Reset gripper
        reset_srv()
        while not gripper_closed:
            rate.sleep()

        rate.sleep()

        S.append(obj_pos)
        L.append(gripper_load)

        rate.sleep()

        if not (len(S) % 5):
            with open('gazebo_start_states.obj', 'wb') as f:
                pickle.dump([S, L], f)
else:
    with open('gazebo_start_states.obj', 'rb') as f:
        S, L = pickle.load(f)


S = np.array(S)
L = np.array(L)

print "Mean: ", np.mean(S, 0), np.mean(L, 0)
print "Std: ", np.std(S, 0), np.std(L, 0)


fig = plt.figure(1)
plt.plot(S[:,0], S[:,1],'.')

fig = plt.figure(2)
plt.plot(L[:,0], L[:,1],'.')

plt.show()


# 200 runs
# Mean:  [3.30313851e-02 1.18306790e+02] [16. 16.]
# Std:  [0.01831497 0.10822673] [0. 0.]



