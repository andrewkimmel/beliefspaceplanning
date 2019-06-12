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


class predict_nn:

    state_action_dim = 6
    state_dim = 4

    VAR_POS = [0.0001,0.0001]# [1.2, 1.2] #[0.8, 0.8]
    VAR_LOAD = [0.0001,0.0001]#[0.7, 0.7] #[0.4, 0.4]
       
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

        with open(data_path + 'delta_mean_arr', 'rb') as pickle_file:
            self.delta_mean_arr = pickle.load(pickle_file)

        with open(data_path + 'delta_std_arr', 'rb') as pickle_file:
            self.delta_std_arr = pickle.load(pickle_file)

        ######## Load Neural Network
        self.model_path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/nn_model/model/'

        '''Neural net structure'''
        self.neural_net_pos = tf.keras.Sequential([
            tfp.layers.DenseFlipout(200, activation=tf.nn.relu,),
            tf.keras.layers.Dropout(rate=0.05),
            tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.05),
            tfp.layers.DenseFlipout(2),
            tf.keras.layers.Dropout(rate=0.05)
        ])

        self.neural_net_load = tf.keras.Sequential([
            tfp.layers.DenseFlipout(200, activation=tf.nn.relu,),
            tf.keras.layers.Dropout(rate=0.05),
            tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.05),
            tfp.layers.DenseFlipout(2),
            tf.keras.layers.Dropout(rate=0.05)
        ])

        self.x = tf.placeholder(tf.float64, shape=[None, self.state_action_dim])
        self.y_pos_mean_pre = self.neural_net_pos(self.x)
        self.y_load_mean_pre = self.neural_net_load(self.x)
        self.y_pos_distribution = tfd.Normal(loc=self.y_pos_mean_pre, scale=self.VAR_POS)
        self.y_load_distribution = tfd.Normal(loc=self.y_load_mean_pre, scale=self.VAR_LOAD)

        self.y_pos_delta_pre = self.y_pos_distribution.sample()
        self.y_load_delta_pre = self.y_load_distribution.sample()


        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init) 

        # Restore variables from disk.
        with self.sess.as_default():
            self.neural_net_pos.load_weights(self.model_path + "d4_s1_pos/BNN_weights")  # load NN parameters
            self.neural_net_load.load_weights(self.model_path + "d4_s1_load/BNN_weights")


    def normalize(self, sa):
        return z_score_normalize(np.asarray([sa]), self.state_mean_arr, self.state_std_arr)

    def denormalize(self, pos_delta, load_delta):
        pos_delta = z_score_denormalize(pos_delta, self.delta_mean_arr, self.delta_std_arr)[0]  
        load_delta = z_score_denormalize(load_delta, self.delta_mean_arr[2:4], self.delta_std_arr[2:4])[0]
        return pos_delta, load_delta

    def predict(self, sa):

        next_input = self.normalize(sa)
        (pos_delta, load_delta) = self.sess.run((self.y_pos_delta_pre, self.y_load_delta_pre), feed_dict={self.x: next_input})
        pos_delta, load_delta = self.denormalize(pos_delta, load_delta)

        next_state = sa[:4] + np.concatenate((pos_delta, load_delta), axis=0)

        return next_state








