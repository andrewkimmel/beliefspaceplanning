""" 
Author: Avishai Sintov
"""
from __future__ import division, print_function, absolute_import

from nn_functions import * # My utility functions

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.io import loadmat
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", help="Retrain existing model", action="store_true")
parser.add_argument("-p", help="Plot trained models", action="store_true")
args = parser.parse_args()
if args.r and args.p:
    training = True
    retrain = True
if args.r:
    training = True
    retrain = True
elif args.p:
    training = False
    retrain = False
else:
    training = True
    retrain = False

DropOut = False
Regularization = False

############################################ Process data ########################################################

print('Loading training data...')

File = 'acrobot_data_cont_v0_d4_m1.mat'
path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/'

Q = loadmat(path + File)
Xt = Q['D']
print( '[NN] Loaded %d transitions.' % Xt.shape[0] )

num_input = 5
num_output = 4

states = Xt[:,:num_input-1]
next_states = Xt[:,num_input:]                                 
actions = Xt[:, num_input-1].reshape(-1,1)

X = np.concatenate((states, actions, next_states), axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
x_mean = scaler.mean_
x_std = scaler.scale_
# X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)

X, Y = X[:,:num_input], X[:,num_input:]

mx = 0.6*np.max(Y, 0)[:2]
mn = 0.6*np.min(Y, 0)[:2]
for i in range(Y.shape[0]):
    for j in range(num_output):
        if j < 2 and np.abs(Y[i,j] - X[i,j]) > np.abs(mx[j]-mn[j]):
            if X[i,j] > mx[j]:
                a =  (Y[i,j] + 1) - X[i,j]
            elif X[i,j] < mn[j]:
                a =  (Y[i,j] - 1) - X[i,j]
            Y[i,j] = a
        else:
            Y[i,j] -= X[i,j] 
# Y -= X[:,:num_output]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

############################################ Network Parameters ########################################################
hidden_layers = [10]*50
activation = 'relu'

# Training Parameters
learning_rate =  0.01
num_steps = int(2e5)
batch_size = 250
display_step = 100

# tf Graph input 
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output])

# Store layers weight & bias
weights, biases = wNb(num_input, hidden_layers, num_output)

# Construct model
keep_prob_input = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
if not DropOut:
    prediction = neural_net(X, weights, biases, activation)
else:
    X_drop = tf.nn.dropout(X, keep_prob=keep_prob_input)
    prediction = neural_net_dropout(X, weights, biases, keep_prob, activation)

# Define loss 
cost = tf.reduce_mean(0.5*tf.pow(prediction - Y, 2))#/(2*n)
# cost = tf.reduce_mean(np.absolute(y_true - y_pred))
# cost = tf.reduce_sum(tf.square(prediction - Y))

# L2 Regularization
if Regularization:
    beta = 0.01
    regularizer = computeReg(weights)
    cost = cost + beta * regularizer

# Define optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdagradOptimizer(learning_rate)
train_op = optimizer.minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

load_from = 'cc_temp.ckpt'
save_to = 'cc_0.ckpt'

# Start Training
# Start a new TF session
COSTS = []	# for plotting
STEPS = []	# for plotting
start = time.time()
with tf.Session() as sess:

    if training:
    
        if not retrain:
            # Run the initializer
            sess.run(init)
        else:
            # Restore variables from disk.
            saver.restore(sess, "./models/" + load_from)                
            print("Loaded saved model: %s" % "./models/" + load_from)

        # Training
        for i in range(1, num_steps+1):
            # Get the next batch 
            batch_x, batch_y = next_batch(batch_size, X_train, Y_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            if not DropOut:
                _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            else:
                _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob_input: 0.5, keep_prob: 0.5})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, c))
                save_path = saver.save(sess, "./models/cc_temp.ckpt")
                COSTS.append(c)
                STEPS.append(i)

        print("Optimization Finished!")

        # Save the variables to disk.
        save_path = saver.save(sess, "./models/" + save_to)
        print("Model saved in path: %s" % save_path)

        # Plot cost convergence
        plt.figure(4)
        # plt.semilogy(STEPS, COSTS, 'k-')
        plt.plot(STEPS, COSTS, 'k-')
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.ylim([0, np.max(COSTS)])
        plt.grid(True)
    else:
        # Restore variables from disk.
        saver.restore(sess, "./models/" + load_from)

    # Testing
    # Calculate cost for training data
    y_train_pred = sess.run(prediction, {X: X_train, keep_prob_input: 1.0, keep_prob: 1.0})
    training_cost = sess.run(cost, feed_dict={X: X_train, Y: Y_train, keep_prob_input: 1.0, keep_prob: 1.0})
    print("Training cost:", training_cost)

    y_test_pred = sess.run(prediction, {X: X_test, keep_prob_input: 1.0, keep_prob: 1.0})
    testing_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test, keep_prob_input: 1.0, keep_prob: 1.0})
    print("Testing cost=", testing_cost)

    tr = '1'
    path = '/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/acrobot_test/cont/'
    action_file = 'acrobot_ao_rrt_plan' + tr + '.txt'
    traj_file = 'acrobot_ao_rrt_traj' + tr + '.txt'
    Ar = np.loadtxt(path + action_file, delimiter=',')
    A = []
    for v in Ar:
        a = v[0]
        n = int(v[1]*100)
        for _ in range(n):
            A.append(a)
    A = np.array(A)[:200].reshape(-1,1)
    S = np.loadtxt(path + traj_file, delimiter=',')[:200,:]
    SA = np.concatenate((S, A, np.zeros((S.shape))), axis=1)
    SA = scaler.transform(SA)
    A = np.copy(SA[:,num_input-1])

    # Closed loop
    s = SA[0,:num_output]
    Spred = s.reshape(1,num_output)
    for i in range(0, A.shape[0]-1):
        # print("[NNCL] Step " + str(i) + " of " + str(A.shape[0]) + ", action: " + str(A[i]))
        a = np.array([A[i]])

        sa = np.concatenate((s, a), axis=0).reshape(1,-1)
        ds = sess.run(prediction, {X: sa, keep_prob_input: 1.0, keep_prob: 1.0})
        s_next = s + ds
        s = np.copy(s_next).reshape((-1,))
        Spred = np.append(Spred, s_next.reshape(1,-1), axis=0)

    Spred = np.concatenate((Spred, A.reshape(-1,1), np.zeros((Spred.shape))), axis=1)
    Spred = scaler.inverse_transform(Spred)
    plt.figure(1)
    ax1 = plt.subplot(1,2,1)
    plt.plot(S[:,0], S[:,1], '.-b', label='rollout mean')
    plt.plot(Spred[:,0], Spred[:,1], '.-k', label='NN')
    plt.legend()
    ax2 = plt.subplot(1,2,2)
    plt.plot(S[:,2], S[:,3], '.-b', label='rollout mean')
    plt.plot(Spred[:,2], Spred[:,3], '.--k', label='NN')

    # Open loop
    plt.figure(2)
    ax1 = plt.subplot(1,2,1)
    plt.plot(S[:,0], S[:,1], '.-b', label='rollout mean')
    ax2 = plt.subplot(1,2,2)
    plt.plot(S[:,2], S[:,3], '.-b', label='rollout mean')
    for i in range(0, A.shape[0]-1):
        # print("[NNOL] Step " + str(i) + " of " + str(A.shape[0]) + ", action: " + str(A[i]))
        a = np.array([A[i]])
        s = SA[i,:num_output]

        sa = np.concatenate((s, a), axis=0).reshape(1,-1)
        ds = sess.run(prediction, {X: sa, keep_prob_input: 1.0, keep_prob: 1.0})
        s_next = s + ds

        Spred = np.concatenate((s.reshape(1,-1), a.reshape(1,-1), s_next.reshape(1,-1)), axis=1)
        Spred = scaler.inverse_transform(Spred)
        ax1 = plt.subplot(1,2,1)
        plt.plot([Spred[0,0], Spred[0,5]], [Spred[0,1], Spred[0,6]], '.-k', label='NN')
        ax2 = plt.subplot(1,2,2)
        plt.plot([Spred[0,2], Spred[0,7]], [Spred[0,3], Spred[0,8]], '.--k', label='NN')

plt.show()


