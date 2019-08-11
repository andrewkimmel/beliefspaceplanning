"""
Bayesian Neural Network Structure
"""

import tensorflow as tf
import tensorflow_probability as tfp
from common.data_normalization import min_max_normalize, z_score_normalize
import numpy as np
import pickle


class BNN:
    def __init__(self, lr=0.001, dropout_p=0.1, batch_size=128, nn_type='0'):
        """
        :param lr: learning rate
        :param dropout_p: dropout probability
        :param batch_size:
        :param nn_type: type 0: 2 hidden layers; type 1: 3 hidden layers
        """
        self.input_dim = None
        self.output_dim = None
        self.lr = lr
        self.dropout_p = dropout_p
        self.neural_net = None
        self.x_data = None  # input data
        self.y_data = None  # target data
        self.training_size = None
        self.held_out_size = None
        self.batch_size = batch_size
        self.var = None
        self.nn_type = nn_type

    def build_neural_net(self):
        """
        Build neural network with input argument(todo)
        """
        if self.nn_type == '0':
            self.neural_net = tf.keras.Sequential([
                tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=self.dropout_p),
                tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=self.dropout_p),
                tfp.layers.DenseFlipout(self.output_dim),
            ])
        elif self.nn_type == '1':
            self.neural_net = tf.keras.Sequential([
                tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=self.dropout_p),
                tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=self.dropout_p),
                tfp.layers.DenseFlipout(256, activation=tf.nn.relu),
                tf.keras.layers.Dropout(rate=self.dropout_p),
                tfp.layers.DenseFlipout(self.output_dim),
            ])

    def add_dataset(self, x_data, y_data, held_out_percentage=0.1):
        """
        Add dataset and get the input and output dimensions
        """
        self.input_dim = x_data.shape[1]
        self.output_dim = y_data.shape[1]
        self.x_data = x_data
        self.y_data = y_data
        self.held_out_size = int(len(x_data) * held_out_percentage)
        self.training_size = len(x_data) - self.held_out_size

    def build_input_pipeline(self,):
        """Build an Iterator switching between train and heldout data. This shuffle part comes from tensorflow tutorial"""
        training_dataset = tf.data.Dataset.from_tensor_slices((self.x_data[:self.training_size], self.y_data[:self.training_size]))
        training_batches = training_dataset.shuffle(self.training_size, reshuffle_each_iteration=True).repeat().batch(
            self.batch_size)
        training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)

        heldout_dataset = tf.data.Dataset.from_tensor_slices\
            ((self.x_data[self.training_size: self.training_size + self.held_out_size],
              self.y_data[self.training_size: self.training_size + self.held_out_size]))
        heldout_frozen = (heldout_dataset.take(self.held_out_size).
                          repeat().batch(self.held_out_size))
        heldout_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_frozen)

        handle = tf.compat.v1.placeholder(tf.string, shape=[])
        feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
            handle, training_batches.output_types, training_batches.output_shapes)
        xs, ys = feedable_iterator.get_next()
        return xs, ys, handle, training_iterator, heldout_iterator

    def train(self, save_path, save_step=10000, var=0.00001, training_step=1000000, normalization=True, normalization_type='z_score', decay='False'
        , load_path=None):
        """
        :param save_path: where to save the weighs and bias as well as normalization parameters
        :param save_step: save model per 500000(default) steps
        :param var: the variance of bayesian nn output, should be trainable(todo)
        :param training_step: maximum training steps
        :param normalization: if normalize data before training
        :param normalization_type: choose 'min_max' or 'z_score' normalization
        :param decay: if decay learning rate while training
        :return:
        """
        if normalization:
            if normalization_type == 'min_max':
                x_min_arr = np.amin(self.x_data, axis=0)
                x_max_arr = np.amax(self.x_data, axis=0)
                y_min_arr = np.amin(self.y_data, axis=0)
                y_max_arr = np.amax(self.y_data, axis=0)
                self.x_data = min_max_normalize(self.x_data, x_min_arr, x_max_arr)
                self.y_data = min_max_normalize(self.y_data, y_min_arr, y_max_arr)
                with open(save_path+'/normalization_arr/normalization_arr', 'wb') as pickle_file:
                    pickle.dump(((x_min_arr, x_max_arr), (y_min_arr, y_max_arr)), pickle_file)
                # with open(save_path+'/normalization_arr/y_normalization_arr', 'wb') as pickle_file:
                #     pickle.dump((y_min_arr, y_max_arr), pickle_file)
            elif normalization_type == 'z_score':
                x_mean_arr = np.mean(self.x_data, axis=0)
                x_std_arr = np.std(self.x_data, axis=0)
                y_mean_arr = np.mean(self.y_data, axis=0)
                y_std_arr = np.std(self.y_data, axis=0)
                self.x_data = z_score_normalize(self.x_data, x_mean_arr, x_std_arr)
                self.y_data = z_score_normalize(self.y_data, y_mean_arr, y_std_arr)
                with open(save_path+'/normalization_arr/normalization_arr', 'wb') as pickle_file:
                    pickle.dump(((x_mean_arr, x_std_arr),(y_mean_arr, y_std_arr)), pickle_file)
                # with open(save_path+'/normalization_arr/y_normalization_arr', 'wb') as pickle_file:
                #     pickle.dump((y_mean_arr, y_std_arr), pickle_file)

        self.var = [var for i in range(self.y_data.shape[1])]  # the variance for bayesian neural network output
        (xs, ys, handle,
         training_iterator, heldout_iterator) = self.build_input_pipeline()
        y_pre = self.neural_net(xs)
        ys_distribution = tfp.distributions.Normal(loc=y_pre, scale=self.var)
        neg_log_likelihood = -tf.reduce_mean(
            input_tensor=ys_distribution.log_prob(ys))
        kl = sum(self.neural_net.losses) / self.batch_size
        elbo_loss = neg_log_likelihood + kl
        predictions = ys_distribution.sample()

        accuracy, accuracy_update_op = tf.metrics.mean_squared_error(
            labels=ys, predictions=predictions)

        with tf.name_scope("train"):
            if decay == 'True': #Add learning rate decay
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(self.lr, global_step, 100000, 0.965, staircase=True)
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate)
                train_op = optimizer.minimize(elbo_loss, global_step=global_step)
            else:
                learning_rate = self.lr
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate)
                train_op = optimizer.minimize(elbo_loss)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            if load_path: 
                print("LOADING WEIGHTS")
                self.neural_net.load_weights(load_path+'/weights/BNN_weights')
            # sess.graph.finalize()
            # Run the training loop.
            train_handle = sess.run(training_iterator.string_handle())
            heldout_handle = sess.run(heldout_iterator.string_handle())

            for step in range(training_step):
                _, _, ac = sess.run([train_op, accuracy_update_op, accuracy],
                                    feed_dict={handle: train_handle})
                if step % 100 == 0:
                    loss_value, accuracy_value = sess.run(
                        [elbo_loss, accuracy], feed_dict={handle: heldout_handle}) #Measure accuracy against heldout data
                    print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.5f}".format(
                        step, loss_value, accuracy_value))
                if step % save_step == 0 and step != 0:
                    print("Saving weights")
                    self.neural_net.save_weights(save_path+'/weights/BNN_weights') #Save weights 

        return accuracy_value





