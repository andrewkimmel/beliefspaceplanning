import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.io
import matplotlib.pyplot as plt
import pickle
import math
from common.data_normalization import z_score_normalize, z_score_denormalize

# config = tf.ConfigProto(
#         device_count={'GPU': 0}
#     )

PRE_START = 0
PRE_END = -1

tf.keras.backend.set_floatx('float64')  # for input weights of NN
tfd = tfp.distributions
TEST_TRAJ_NUM = 1

''' load mean and std used for normalization'''
with open('../common/acrobot/normalization_arr_without_noisy/state_mean_arr', 'rb') as pickle_file:
    state_mean_arr = pickle.load(pickle_file)

with open('../common/acrobot/normalization_arr_without_noisy/state_std_arr', 'rb') as pickle_file:
    state_std_arr = pickle.load(pickle_file)

with open('../common/acrobot/normalization_arr_without_noisy/d_ang_mean_arr', 'rb') as pickle_file:
    d_ang_mean_arr = pickle.load(pickle_file)

with open('../common/acrobot/normalization_arr_without_noisy/d_ang_std_arr', 'rb') as pickle_file:
    d_ang_std_arr = pickle.load(pickle_file)

with open('../common/acrobot/normalization_arr_without_noisy/d_vel_mean_arr', 'rb') as pickle_file:
    d_vel_mean_arr = pickle.load(pickle_file)

with open('../common/acrobot/normalization_arr_without_noisy/d_vel_std_arr', 'rb') as pickle_file:
    d_vel_std_arr = pickle.load(pickle_file)

with open('../test_data/acrobot_ao_rrt_traj' + str(TEST_TRAJ_NUM), 'rb') as pickle_file:
    gt = pickle.load(pickle_file)

if PRE_END < 0:
    gt = np.asarray(gt)[PRE_START:PRE_END]
else:
    gt = np.asarray(gt)[PRE_START:PRE_END+1]


with open('../test_data//acrobot_ao_rrt_plan' + str(TEST_TRAJ_NUM), 'rb') as pickle_file:
    acts = pickle.load(pickle_file)

PRE_STEP = len(gt) - 2


def wrap_acrobot_action(plan_act):
    act_seq = []
    for act_pair in plan_act:
        a = act_pair[0]
        apply_num = int(act_pair[1] * 100)
        for i in range(apply_num):
            act_seq.append(a)
    return act_seq


def map_angle(state):
    if state[0] > math.pi:
        print(state)
        state[0] = - math.pi + (state[0] - math.pi)
        print(state)
    if state[0] < -math.pi:
        state[0] = state[0] + 2 * math.pi
    if state[1] > math.pi:
        print(state)
        state[1] = - math.pi + (state[1] - math.pi)
        print(state)
    if state[1] < -math.pi:
        state[1] = state[1] + 2 * math.pi
    return state


acts = wrap_acrobot_action(acts)[PRE_START:PRE_END]
validation_data = np.asarray(gt[:-1])
validation_data = np.append(validation_data, np.asarray(acts).reshape((len(validation_data), 1)), axis=1)

VAR_ANG = [0.0000001, 0.0000001]
VAR_VEL = [0.0000001, 0.0000001]

'''Neural net structure'''
neural_net_ang = tf.keras.Sequential([
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu,),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(200, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.05),
    tfp.layers.DenseFlipout(2),
    # tf.keras.layers.Dropout(rate=0.05)
])

neural_net_vel = tf.keras.Sequential([
    tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.1),
    tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.1),
    tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.1),
    tfp.layers.DenseFlipout(2),
    # tf.keras.layers.Dropout(rate=DROPOUT_P)
])

x = tf.placeholder(tf.float64, shape=[None, 5])
y_ang_mean_pre = neural_net_ang(x)
y_vel_mean_pre = neural_net_vel(x)

y_ang_distribution = tfd.Normal(loc=y_ang_mean_pre, scale=VAR_ANG)
y_vel_distribution = tfd.Normal(loc=y_vel_mean_pre, scale=VAR_VEL)

y_ang_delta_pre = y_ang_distribution.sample()
y_vel_delta_pre = y_vel_distribution.sample()


with tf.Session() as sess:

    pre_trajectory_vel = []
    pre_trajectory_ang = []
    init = tf.global_variables_initializer()
    sess.run(init)

    neural_net_ang.load_weights("../NN_model/acrobot/d4_s1_ang/BNN_weights")  # load NN parameters
    neural_net_vel.load_weights("../NN_model/acrobot/d4_s1_vel/BNN_weights")

    pre_trajectory_vel.append(validation_data[0][2:4])
    pre_trajectory_ang.append(validation_data[0][:2])
    next_state = validation_data[0]
    next_input = np.asarray([next_state])

    for i in range(PRE_STEP):
        next_input = z_score_normalize(next_input, state_mean_arr, state_std_arr)
        (ang_delta, vel_delta) = sess.run((y_ang_delta_pre, y_vel_delta_pre), feed_dict={x: next_input})
        ang_delta = z_score_denormalize(ang_delta, d_ang_mean_arr, d_ang_std_arr)[0]  # denormalize
        vel_delta = z_score_denormalize(vel_delta, d_vel_mean_arr, d_vel_std_arr)[0]

        next_vel = next_state[2:4] + vel_delta
        next_ang = next_state[:2] + ang_delta

        next_state = np.append(next_state[:4] + np.concatenate((ang_delta, vel_delta), axis=0),
                               validation_data[i + 1][4:5])


        pre_trajectory_vel.append(next_vel)
        pre_trajectory_ang.append(next_ang)
        next_state = map_angle(next_state)
        next_input = np.asarray([next_state])
        last_vel_delta = vel_delta

pre_trajectory_vel = np.asarray(pre_trajectory_vel)
pre_trajectory_ang = np.asarray(pre_trajectory_ang)

plt.figure(1)
plt.scatter(gt[0, 2], gt[0, 3], marker="*", label='start')
plt.plot(gt[:, 2], gt[:, 3], color='blue', label='Ground Truth', marker='.')
plt.plot(pre_trajectory_vel[:, 0], pre_trajectory_vel[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- Velocity Space')
plt.legend()
plt.show()
# plt.savefig('../../../plots/acrobot/prediction/vel/path' + str(TEST_TRAJ_NUM) + '_' + str(PRE_START) + '_' + str(PRE_END))

plt.figure(2)
plt.scatter(gt[0, 0], gt[0, 1], marker="*", label='start')
plt.plot(gt[:, 0], gt[:, 1], color='blue', label='Ground Truth', marker='.')
plt.plot(pre_trajectory_ang[:, 0], pre_trajectory_ang[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Bayesian NN Prediction -- angle Space')
plt.legend()
plt.show()
# plt.savefig('../../../plots/acrobot/prediction/ang/path' + str(TEST_TRAJ_NUM) + '_' + str(PRE_START) + '_' + str(PRE_END))
