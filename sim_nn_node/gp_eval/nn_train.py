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
import pickle

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

print('Loading training data...')

l_prior = 40
dim_ = 1
path = '/home/juntao/catkin_ws/src/beliefspaceplanning/sim_nn_node/gp_eval/'
with open(path + 'error_points_P' + str(l_prior) + '.pkl', 'rb') as f: 
    O, L, Apr, E = pickle.load(f, encoding='bytes')
X = []
for o, apr, l, e in zip(O, Apr, L, E):
    x = np.concatenate((o[:6], np.array([l]), apr.reshape((-1)), np.array([e])), axis = 0)
    X.append(x)
X = np.array(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
x_mean = scaler.mean_
x_std = scaler.scale_
# X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)

X, Y = X[:,:-1], X[:,-1].reshape(-1,1)
state_action_dim = X.shape[1]
num_output = 1

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state= 42)

# Network Parameters
hidden_layers = [20]*5
activation = 2

# Training Parameters
learning_rate =  0.1
num_steps = int(1e4)
batch_size = 150
display_step = 100

# tf Graph input 
X = tf.placeholder("float", [None, state_action_dim])
Y = tf.placeholder("float", [None, num_output])

# Store layers weight & bias
weights, biases = wNb(state_action_dim, hidden_layers, num_output)

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
optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdagradOptimizer(learning_rate)
train_op = optimizer.minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
model_file = path + '/models/' + 'nn' + '.ckpt'

# Start Training
# Start a new TF session
COSTS = []	# for plotting
STEPS = []	# for plotting
start = time.time()
with tf.Session() as sess:

    if training:
    
        if  not retrain:
            # Run the initializer
            sess.run(init)
        else:
            # Restore variables from disk.
            saver.restore(sess, model_file)                
            # print("Loaded saved model: %s" % "./models/" + load_from)

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
                save_path = saver.save(sess, model_file)
                COSTS.append(c)
                STEPS.append(i)

        print("Optimization Finished!")

        # Save the variables to disk.
        save_path = saver.save(sess, model_file)
        print("Model saved in path: %s" % save_path)

        # Plot cost convergence
        plt.figure(4)
        plt.semilogy(STEPS, COSTS, 'k-')
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.ylim([0, np.max(COSTS)])
        plt.grid(True)
    else:
        # Restore variables from disk.
        saver.restore(sess, model_file)

    # Testing
    # Calculate cost for training data
    Y_train_pred = sess.run(prediction, {X: X_train, keep_prob_input: 1.0, keep_prob: 1.0})
    training_cost = sess.run(cost, feed_dict={X: X_train, Y: Y_train, keep_prob_input: 1.0, keep_prob: 1.0})
    print("Training cost:", training_cost)

    Y_test_pred = sess.run(prediction, {X: X_test, keep_prob_input: 1.0, keep_prob: 1.0})
    testing_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test, keep_prob_input: 1.0, keep_prob: 1.0})
    print("Testing cost=", testing_cost)

    

plt.show()


