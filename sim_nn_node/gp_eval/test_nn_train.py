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

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# x_mean = scaler.mean_
# x_std = scaler.scale_
# X = X*x_std + x_mean # Denormalize or use scaler.inverse_transform(X)

X, Y = X[:,:-1], X[:,-1].reshape(-1,1)
state_action_dim = X.shape[1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state= 42)

# Define the number of nodes
n_nodes_hl1 = 30
n_nodes_hl2 = 30

# Define the number of outputs and the learn rate
n_classes = 1
learn_rate = 0.05

# Define input / output placeholders
x_ph = tf.placeholder('float', [None, state_action_dim])
y_ph = tf.placeholder('float')

model_file = path + '/models/' + 'nn' + '.ckpt'

# Routine to compute the neural network (2 hidden layers)
def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([state_action_dim, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

    return output


# Routine to train the neural network
def train_neural_network(x_ph):
    prediction = neural_network_model(x_ph)
    cost = tf.reduce_mean(tf.square(prediction - y_ph))
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)

    # cycles feed forward + backprop
    hm_epochs = 1000000

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    COSTS = []	# for plotting
    STEPS = []	# for plotting
    t = 0
    with tf.Session() as sess:
        if 1:
            # Run the initializer
            sess.run(tf.global_variables_initializer())
        else:
            # Restore variables from disk.
            saver.restore(sess, model_file)                
            print("Loaded saved model: %s" % model_file)

        # Train in each epoch with the whole data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict = {x_ph: X_train, y_ph: Y_train})
            epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            if epoch % 100 == 0:
                save_path = saver.save(sess, model_file)
                COSTS.append(epoch_loss)
                STEPS.append(epoch)

            if epoch_loss < 1e-1:
                t += 1
                if t > 8:
                    break
            else:
                t = 0

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy;', accuracy.eval({x_ph: X_test, y_ph: Y_test}))

        plt.figure(0)
        plt.plot(STEPS, COSTS, 'k-')
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.ylim([0, np.max(COSTS)])
        plt.grid(True)
        plt.show()

def test_nn():

    prediction = neural_network_model(x_ph)
    cost = tf.reduce_mean(tf.square(prediction - y_ph))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_file)                
        print("Loaded saved model: %s" % model_file)

        Err = []
        T = []
        for x_test, y_test in zip(X_test, Y_test):
            st = time.time()
            e_pred = sess.run(prediction, feed_dict = {x_ph: x_test.reshape(1,-1)})[0]
            T.append(time.time() - st)
            print (e_pred, y_test)     
            Err.append(np.abs(y_test-e_pred))   

    print ("Time: " + str(np.mean(T)))
    print ("Error: " + str(np.mean(Err)))

# Train network
# train_neural_network(x_ph)

test_nn()



    
