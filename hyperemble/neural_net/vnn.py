from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import math
import numpy as np
import tensorflow as tf

from abc import ABCMeta
from sklearn.base import BaseEstimator
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score, accuracy_score
from keras.datasets import mnist

from hyperemble.utils import mean_squared_error, ProcessBatch, PrintMess


def _add_layer(inputs, input_dim, output_dim, keep_prob, scope=None):
    with tf.name_scope(scope):
        weights = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                              stddev=1. / math.sqrt(float(input_dim))),
                              name='weights')
        biases = tf.Variable(tf.zeros([output_dim]), name='biases')
        output_data = tf.matmul(inputs, weights) + biases
        return tf.nn.dropout(tf.nn.relu(output_data), keep_prob=keep_prob)


def _add_multi_layer(inputs, input_dim, hidden_dim, n_layers,
                     keep_prob):
    res = _add_layer(inputs, input_dim, hidden_dim, keep_prob, scope="layer0")
    for i in range(1, n_layers):
        res = _add_layer(res, hidden_dim, hidden_dim, keep_prob,
                         scope="layer{0}".format(i))
    return res


class VanillaNeuralNet(six.with_metaclass(ABCMeta, BaseEstimator)):
    loss_functions = {
        "mse": mean_squared_error,
        "softmax": tf.nn.sparse_softmax_cross_entropy_with_logits,
    }

    def __init__(self, n_layers, hidden_dim, keep_prob, n_iter=5,
                 learning_rate=0.001, loss_func="auto",
                 verbose=1, batch_size=128, random_state=None):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.verbose = verbose
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.random_state = random_state
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.sess = tf.Session()

    def set_params(self, *args, **kwargs):
        super(VanillaNeuralNet, self).set_params(*args, **kwargs)
        return self

    def _construct_graph(self, input_dim, output_dim):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.set_random_seed(self.random_state)
        self.x_pl = tf.placeholder(tf.float32, [None, input_dim])
        if self.loss_func == "softmax":
            self.y_pl = tf.placeholder(tf.int64, [None])
        elif self.loss_func == "mse":
            self.y_pl = tf.placeholder(tf.float32, [None, output_dim])
        self.keep_prob_pl = tf.placeholder(tf.float32)

        outputs = _add_multi_layer(self.x_pl,
                                   input_dim=input_dim,
                                   hidden_dim=self.hidden_dim,
                                   n_layers=self.n_layers,
                                   keep_prob=self.keep_prob)
        softmax_w = tf.Variable(tf.truncated_normal(
                                [self.hidden_dim, output_dim],
                                stddev=1. / math.sqrt(float(input_dim))),
                                name='softmax_w')
        softmax_b = tf.Variable(tf.zeros([output_dim]), name='softmax_b')
        self.logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.loss = self.loss_functions[self.loss_func](self.logits, self.y_pl)
        if self.loss_func == "softmax":
            self.loss = tf.reduce_mean(self.loss)
        self.update = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

    def step(self, x, y):
        feed_in = {self.x_pl: x, self.y_pl: y,
                   self.keep_prob_pl: self.keep_prob}
        feed_out = [self.loss, self.update]
        return self.sess.run(feed_out, feed_dict=feed_in)

    def fit(self, x, y, validation_proportion=0.1):
        if self.loss_func == "auto":
            if y.dtype == np.float:
                self.loss_func = "mse"
            else:
                self.loss_func = "softmax"
        if self.loss_func == "mse":
            if len(y.shape) == 1:
                y = y.reshape((-1, 1))
            self.output_dim = y.shape[1]
        elif self.loss_func == "softmax":
            self.output_dim = np.max(y) + 1

        n_obs, self.n_features = x.shape
        rs = ShuffleSplit(n_obs, n_iter=1, test_size=validation_proportion,
                          random_state=self.random_state)
        for train_index, valid_index in rs:
            pass
        df = ProcessBatch(x[train_index], y[train_index])
        self._construct_graph(self.n_features, self.output_dim)
        self.sess.run(tf.initialize_all_variables())
        logger = PrintMess()
        if self.verbose:
            logger.info(header=True, Iter=0, TrnLoss=0, ValScore=0)
        for i in range(int(self.n_iter * n_obs / self.batch_size)):
            x_batch, y_batch = df.next_batch(self.batch_size)
            res = self.step(x_batch, y_batch)
            if (i % 40 == 0) and self.verbose:
                score = self.score(x[valid_index], y[valid_index])
                logger.info(header=False, Iter=i, TrnLoss=res[0],
                            ValScore=score)

    def predict(self, x):
        feed_in = {self.x_pl: x, self.keep_prob_pl: 1.}
        feed_out = [self.logits]
        yhat = self.sess.run(feed_out, feed_dict=feed_in)[0]
        if self.loss_func == "softmax":
            yhat = np.argmax(yhat, axis=1)
        return yhat

    def score(self, x, y):
        yhat = self.predict(x)
        if self.loss_func == "mse":
            if self.output_dim == 1:
                return r2_score(y, yhat[:, 0])
            else:
                return np.mean([r2_score(y[:, i], yhat[:, i]) for i in
                                range(self.output_dim)])
        else:
            return accuracy_score(y, yhat)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    clf = VanillaNeuralNet(n_layers=2, hidden_dim=200,
                           keep_prob=0.8, loss_func="auto",
                           verbose=1, batch_size=128, random_state=1)
    clf.fit(X_train, y_train)
