""" 
Author: Avishai Sintov
"""
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt
import time
from common.data_normalization import z_score_normalize, z_score_denormalize
import pickle
from keras import backend as K
import math


class predict_nn:

    state_action_dim = 5
    state_dim = 4

    VAR_ang = [0.00001, 0.00001]
    VAR_vel = [0.00001, 0.00001]
       
    def __init__(self):

        tf.keras.backend.set_floatx('float64')  # for input weights of NN
        tfd = tfp.distributions

        ####### Load training data for failure check
        print('[predict_nn] Loading training data...')

        ''' load mean and std used for normalization'''
        data_path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/nn_model/data/'
        with open(data_path + 'state_mean_arr', 'rb') as pickle_file:
            self.state_mean_arr = pickle.load(pickle_file)

        with open(data_path + 'state_std_arr', 'rb') as pickle_file:
            self.state_std_arr = pickle.load(pickle_file)

        with open(data_path + 'd_ang_mean_arr', 'rb') as pickle_file:
            self.d_ang_mean_arr = pickle.load(pickle_file)

        with open(data_path + 'd_ang_std_arr', 'rb') as pickle_file:
            self.d_ang_std_arr = pickle.load(pickle_file)

        with open(data_path + 'd_vel_mean_arr', 'rb') as pickle_file:
            self.d_vel_mean_arr = pickle.load(pickle_file)

        with open(data_path + 'd_vel_std_arr', 'rb') as pickle_file:
            self.d_vel_std_arr = pickle.load(pickle_file)

        ######## Load Neural Network
        self.model_path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/nn_model/model/'

        '''Neural net structure'''
        self.neural_net_ang = tf.keras.Sequential([
            tfp.layers.DenseFlipout(200, activation=tf.nn.relu,),
            tf.keras.layers.Dropout(rate=0.05),
            tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.05),
            tfp.layers.DenseFlipout(2),
            # tf.keras.layers.Dropout(rate=0.05)
        ])

        self.neural_net_vel = tf.keras.Sequential([
            tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tfp.layers.DenseFlipout(2),
            # tf.keras.layers.Dropout(rate=DROPOUT_P)
        ])

        self.x = tf.placeholder(tf.float64, shape=[None, self.state_action_dim])
        self.y_ang_mean_pre = self.neural_net_ang(self.x)
        self.y_vel_mean_pre = self.neural_net_vel(self.x)
        self.y_ang_distribution = tfd.Normal(loc=self.y_ang_mean_pre, scale=self.VAR_ang)
        self.y_vel_distribution = tfd.Normal(loc=self.y_vel_mean_pre, scale=self.VAR_vel)

        self.y_ang_delta_pre = self.y_ang_distribution.sample()
        self.y_vel_delta_pre = self.y_vel_distribution.sample()


        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init) 

        # Restore variables from disk.
        with self.sess.as_default():
            self.neural_net_ang.load_weights(self.model_path + "d4_s1_ang/BNN_weights")  # load NN parameters
            self.neural_net_vel.load_weights(self.model_path + "d4_s1_vel/BNN_weights")

    def map_angle(self, state):
        if state[0] > math.pi:
            state[0] = - math.pi + (state[0] - math.pi)
        if state[0] < -math.pi:
            state[0] = state[0] + 2 * math.pi
        if state[1] > math.pi:
            state[1] = - math.pi + (state[1] - math.pi)
        if state[1] < -math.pi:
            state[1] = state[1] + 2 * math.pi
        return state

    def normalize(self, sa):
        return z_score_normalize(np.asarray([sa]), self.state_mean_arr, self.state_std_arr)

    def denormalize(self, ang_delta, vel_delta):
        ang_delta = z_score_denormalize(ang_delta, self.d_ang_mean_arr, self.d_ang_std_arr)[0]  
        vel_delta = z_score_denormalize(vel_delta, self.d_vel_mean_arr, self.d_vel_std_arr)[0]
        return ang_delta, vel_delta

    def predict(self, sa):

        next_input = self.normalize(sa)
        (ang_delta, vel_delta) = self.sess.run((self.y_ang_delta_pre, self.y_vel_delta_pre), feed_dict={self.x: next_input})
        ang_delta, vel_delta = self.denormalize(ang_delta, vel_delta)

        next_state = sa[:4] + np.concatenate((ang_delta, vel_delta), axis=0)
        next_state = self.map_angle(next_state)

        return next_state

# if __name__ == "__main__":
#     NN = predict_nn()

#     with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/nn_model/data/acrobot_ao_rrt_plan1', 'rb') as pickle_file:
#         A = pickle.load(pickle_file, encoding='latin1')
#     with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/nn_model/data/acrobot_ao_rrt_traj1', 'rb') as pickle_file:
#         S = pickle.load(pickle_file, encoding='latin1')
#     S = np.asarray(S)
#     An = np.asarray(A)

#     A = []
#     for v in An:
#         for i in range(int(v[1]*100)):
#             A.append(v[0])
#     A = np.array(A)

#     s = S[0,:]
#     SP = []
#     SP.append(s)
#     t = 0.
#     k = 0
#     for a in A:
#         for i in range(1):
#             print(k, i, a)
#             sa = np.append(s, a)
#             # sa = np.concatenate((s, a), axis=0)
#             st = time.time()
#             s_next = NN.predict(sa)
#             t += time.time() - st
#             k += 1

#             SP.append(s_next)
#             s = np.copy(s_next)

#     print("Avg. time: %f"%(t/k))

#     SP = np.array(SP)

#     plt.figure(1)
#     plt.plot(S[:,0], S[:,1], 'r')
#     plt.plot(SP[:,0],SP[:,1], '.-b')
#     plt.figure(2)
#     plt.plot(S[:,2], S[:,3], 'r')
#     plt.plot(SP[:,2],SP[:,3], '.-b')
#     plt.show()







