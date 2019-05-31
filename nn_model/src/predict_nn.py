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
        data_path = '/home/juntao/catkin_ws/src/beliefspaceplanning/nn_model/data/'
        with open(data_path + 'state_mean_arr', 'rb') as pickle_file:
            self.state_mean_arr = pickle.load(pickle_file)

        with open(data_path + 'state_std_arr', 'rb') as pickle_file:
            self.state_std_arr = pickle.load(pickle_file)

        with open(data_path + 'delta_mean_arr', 'rb') as pickle_file:
            self.delta_mean_arr = pickle.load(pickle_file)

        with open(data_path + 'delta_std_arr', 'rb') as pickle_file:
            self.delta_std_arr = pickle.load(pickle_file)

        ######## Load Neural Network
        self.model_path = '/home/juntao/catkin_ws/src/beliefspaceplanning/nn_model/model/'

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

# if __name__ == "__main__":
#     NN = predict_nn()

#     # with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/jt_path3_v14_m10.pkl', 'rb') as pickle_file:
#     #     traj_data = pickle.load(pickle_file, encoding='latin1')
#     # S = np.asarray(traj_data[0])#[:-1,:]
#     # A = traj_data[2]
#     with open('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/src/results/jt_rollout_3_v14_d8_m1.pkl', 'rb') as pickle_file:
#         traj_data = pickle.load(pickle_file, encoding='latin1')
#     S = np.asarray(traj_data[0])[:,:4]
#     A = traj_data[1]

#     s = S[0,:]
#     SP = []
#     SP.append(s)
#     t = 0.
#     k = 0
#     for a in A:
#         for i in range(1):
#             print(k, i, a)
#             sa = np.concatenate((s, a), axis=0)
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







