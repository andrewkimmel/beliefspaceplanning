import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.io
import matplotlib.pyplot as plt
import pickle
from data_normalization import z_score_normalize, z_score_denormalize
import time


tf.keras.backend.set_floatx('float64')  # for input weights of NN
tfd = tfp.distributions

''' load mean and std used for normalization'''
with open('../data/state_mean_arr', 'rb') as pickle_file:
    state_mean_arr = pickle.load(pickle_file)

with open('../data/state_std_arr', 'rb') as pickle_file:
    state_std_arr = pickle.load(pickle_file)

with open('../data/delta_mean_arr', 'rb') as pickle_file:
    delta_mean_arr = pickle.load(pickle_file)

with open('../data/delta_std_arr', 'rb') as pickle_file:
    delta_std_arr = pickle.load(pickle_file)

with open('../data/jt_path2_v14_m10.pkl', 'rb') as pickle_file:
    traj_data = pickle.load(pickle_file, encoding='latin1')

gt = traj_data[0]
gp_pre = traj_data[1]
acts = traj_data[2]

validation_data = np.asarray(gt[:-1])
validation_data = np.append(validation_data, np.asarray(acts), axis=1)
# validation_data = z_score_normalize(validation_data, mean_arr, std_arr)

VAR_POS = [0.00001, 0.00001]
VAR_LOAD = [0.00001, 0.00001]
PRE_STEP = len(gt) - 2  # the number of steps to predict

'''Neural net structure'''
neural_net_pos = tf.keras.Sequential([
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu,),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(2),
    tf.keras.layers.Dropout(rate=0.05)
])

neural_net_load = tf.keras.Sequential([
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu,),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(2),
    tf.keras.layers.Dropout(rate=0.05)
])

x = tf.placeholder(tf.float64, shape=[None, 6])
y_pos_mean_pre = neural_net_pos(x)
y_load_mean_pre = neural_net_load(x)
y_pos_distribution = tfd.Normal(loc=y_pos_mean_pre, scale=VAR_POS)
y_load_distribution = tfd.Normal(loc=y_load_mean_pre, scale=VAR_LOAD)

y_pos_delta_pre = y_pos_distribution.sample()
y_load_delta_pre = y_load_distribution.sample()

with tf.Session() as sess:
    pre_trajectory = []
    init = tf.global_variables_initializer()
    sess.run(init)
    neural_net_pos.load_weights("../model/d4_s1_pos/BNN_weights")  # load NN parameters   Model load should after session initial.
    neural_net_load.load_weights("../model/d4_s1_load/BNN_weights")
    pre_trajectory.append(validation_data[0][:2])
    next_state = validation_data[0]
    print('s', next_state)
    next_input = z_score_normalize(np.asarray([next_state]), state_mean_arr, state_std_arr)
    # print('next', next_input)
    for i in range(PRE_STEP):
        for _ in range(10):
            # st = time.time()
            (pos_delta, load_delta) = sess.run((y_pos_delta_pre, y_load_delta_pre), feed_dict={x: next_input})
            # print('1', pos_delta, load_delta)
            pos_delta = z_score_denormalize(pos_delta, delta_mean_arr, delta_std_arr)[0]  # denormalize
            load_delta = z_score_denormalize(load_delta, delta_mean_arr[2:4], delta_std_arr[2:4])[0]
            # print(pos_delta, load_delta)
            next_pos = next_state[0:2] + pos_delta
            pre_trajectory.append(next_pos)
            next_state = np.append(next_state[:4] + np.concatenate((pos_delta, load_delta), axis=0), validation_data[i + 1][4:6])
            next_input = z_score_normalize(np.asarray([next_state]), state_mean_arr, state_std_arr)
            print('sn', next_state)
            # print(time.time() - st)



pre_trajectory = np.asarray(pre_trajectory)

plt.figure(1)
plt.scatter(gt[0, 0], gt[0, 1], marker="*", label='start')
plt.plot(gt[:, 0], gt[:, 1], color='blue', label='Ground Truth')
plt.plot(pre_trajectory[:, 0], pre_trajectory[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- Pos Space')
plt.legend()
plt.show()
