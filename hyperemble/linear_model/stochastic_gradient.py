from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score, accuracy_score

from hyperemble.utils import (ProcessBatch, mean_squared_error,
                              mean_absolute_deviation, PrintMess,
                              logistic_loss, hinge_loss, exp_loss,
                              squared_hinge_loss)


class BaseSGD(six.with_metaclass(ABCMeta, BaseEstimator)):
    def __init__(self, loss_name, l1_penalty=0.0001,
                 l2_penalty=0.0001, fit_intercept=True, n_iter=5,
                 verbose=0, random_state=None, learning_rate=0.001,
                 warm_start=False, batch_size=128):
        self.loss_name = loss_name
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.batch_size = batch_size

    def set_params(self, *args, **kwargs):
        super(BaseSGD, self).set_params(*args, **kwargs)
        return self

    @abstractmethod
    def fit(self, x, y):
        """Fit model."""

    @abstractmethod
    def step(self, x, y):
        """Make one stochastic gradient step"""

    @abstractmethod
    def predict(self, x):
        """Make prediction"""

    @abstractmethod
    def score(self, x, y):
        """Return score of prediction"""


class SGDRegressor(BaseSGD):
    loss_functions = {
        "mse": mean_squared_error,
        "squared_error": mean_squared_error,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_deviation": mean_absolute_deviation,
        "mad": mean_absolute_deviation,
    }

    def __init__(self, loss_name="squared_loss", l1_penalty=0.0001,
                 l2_penalty=0.0001, fit_intercept=True, n_iter=5,
                 verbose=0, random_state=None, learning_rate=0.01,
                 warm_start=False, batch_size=128):
        super(SGDRegressor, self).__init__(loss_name=loss_name,
                                           l1_penalty=l1_penalty,
                                           l2_penalty=l2_penalty,
                                           fit_intercept=fit_intercept,
                                           n_iter=n_iter,
                                           verbose=verbose,
                                           random_state=random_state,
                                           learning_rate=learning_rate,
                                           warm_start=warm_start,
                                           batch_size=batch_size)
        self.sess = tf.Session()

    def _construct_graph(self, input_dim, output_dim):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.set_random_seed(self.random_state)
        self.x_pl = tf.placeholder(tf.float32, [None, input_dim])
        self.y_pl = tf.placeholder(tf.float32, [None, output_dim])
        W = tf.Variable(tf.random_normal([input_dim, output_dim]))
        b = tf.Variable(tf.zeros([output_dim]))
        if self.fit_intercept:
            self.yhat = tf.matmul(self.x_pl, W) + b
        else:
            self.yhat = tf.matmul(self.x_pl, W)
        self.loss = self.loss_functions[self.loss_name](self.y_pl, self.yhat)\
            + self.l2_penalty * tf.reduce_sum(tf.square(W))\
            + self.l1_penalty * tf.reduce_sum(tf.abs(W))
        self.update = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

    def step(self, x, y):
        feed_in = {self.x_pl: x, self.y_pl: y}
        feed_out = [self.loss, self.update]
        return self.sess.run(feed_out, feed_dict=feed_in)

    def fit(self, x, y, validation_proportion=0.1):
        n, self.p = x.shape
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        _, self.q = y.shape
        rs = ShuffleSplit(n, n_iter=1, test_size=validation_proportion,
                          random_state=self.random_state)
        for train_index, valid_index in rs:
            pass
        df = ProcessBatch(x[train_index], y[train_index])
        self._construct_graph(self.p, self.q)
        self.sess.run(tf.initialize_all_variables())
        logger = PrintMess()
        if self.verbose:
            logger.info(header=True, Iter=0, TrnLoss=0, ValScore=0)
        for i in range(int(self.n_iter * n / self.batch_size)):
            x_batch, y_batch = df.next_batch(self.batch_size)
            res = self.step(x_batch, y_batch)
            if (i % 40 == 0) and self.verbose:
                yhat = self.predict(x[valid_index])
                score = r2_score(y[valid_index], yhat)
                logger.info(header=False, Iter=i, TrnLoss=res[0],
                            ValScore=score)

    def predict(self, x):
        feed_in = {self.x_pl: x}
        feed_out = [self.yhat]
        yhat = self.sess.run(feed_out, feed_dict=feed_in)[0]
        if self.q == 1:
            yhat = yhat.ravel()
        return yhat

    def score(self, x, y):
        yhat = self.predict(x)
        return r2_score(y, yhat)


class SGDClassifier(BaseSGD):
    loss_functions = {
        "log": logistic_loss,
        "hinge": hinge_loss,
        "exp": exp_loss,
        "squared_hinge": squared_hinge_loss,
        "mse": mean_squared_error,
        "mad": mean_absolute_deviation,
    }

    def __init__(self, loss_name="log", l1_penalty=0.0001,
                 l2_penalty=0.0001, fit_intercept=True, n_iter=5,
                 verbose=0, random_state=None, learning_rate=0.01,
                 warm_start=False, batch_size=128):
        super(SGDClassifier, self).__init__(loss_name=loss_name,
                                            l1_penalty=l1_penalty,
                                            l2_penalty=l2_penalty,
                                            fit_intercept=fit_intercept,
                                            n_iter=n_iter,
                                            verbose=verbose,
                                            random_state=random_state,
                                            learning_rate=learning_rate,
                                            warm_start=warm_start,
                                            batch_size=batch_size)
        self.sess = tf.Session()

    def _construct_graph(self, input_dim):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.set_random_seed(self.random_state)
        self.x_pl = tf.placeholder(tf.float32, [None, input_dim])
        self.y_pl = tf.placeholder(tf.float32, [None])
        W = tf.Variable(tf.random_normal([input_dim, 1]))
        b = tf.Variable(tf.zeros([1]))
        self.yhat = tf.reshape(tf.matmul(self.x_pl, W), (-1,))
        if self.fit_intercept:
            self.yhat += b
        self.loss = self.loss_functions[self.loss_name](self.y_pl, self.yhat)\
            + self.l2_penalty * tf.reduce_sum(tf.square(W))\
            + self.l1_penalty * tf.reduce_sum(tf.abs(W))
        self.update = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

    def step(self, x, y):
        feed_in = {self.x_pl: x, self.y_pl: y}
        feed_out = [self.loss, self.update]
        return self.sess.run(feed_out, feed_dict=feed_in)

    def binarize(self, y):
        self.classes = np.unique(y)
        assert len(self.classes) == 2
        res = np.ones(len(y), dtype=np.float32)
        res[y == self.classes[0]] = -1
        return res

    def unbinarize(self, y):
        res = np.zeros(len(y), dtype=np.int64)
        res[y == 1] = self.classes[1]
        res[y == -1] = self.classes[0]
        return res

    def fit(self, x, y, validation_proportion=0.1):
        n, self.p = x.shape
        y_new = self.binarize(y)

        rs = ShuffleSplit(n, n_iter=1, test_size=validation_proportion,
                          random_state=self.random_state)
        for train_index, valid_index in rs:
            pass
        df = ProcessBatch(x[train_index], y_new[train_index])
        self._construct_graph(self.p)
        self.sess.run(tf.initialize_all_variables())
        logger = PrintMess()
        if self.verbose:
            logger.info(header=True, Iter=0, TrnLoss=0, ValScore=0)
        for i in range(int(self.n_iter * n / self.batch_size)):
            x_batch, y_batch = df.next_batch(self.batch_size)
            res = self.step(x_batch, y_batch)
            if (i % 40 == 0) and self.verbose:
                yhat = self.predict(x[valid_index])
                score = accuracy_score(y[valid_index], yhat)
                logger.info(header=False, Iter=i, TrnLoss=res[0],
                            ValScore=score)

    def predict(self, x):
        feed_in = {self.x_pl: x}
        feed_out = [self.yhat]
        yhat = self.sess.run(feed_out, feed_dict=feed_in)[0]
        res = np.ones(len(x), dtype=np.int64)
        res[yhat < 0] = -1
        return self.unbinarize(res)

    def score(self, x, y):
        yhat = self.predict(x)
        return accuracy_score(y, yhat)


class ElasticNet(SGDRegressor):
    def __init__(self, fit_intercept=True, l1_penalty=.0001, l2_penalty=0.0001,
                 n_iter=5, verbose=0, random_state=None,
                 learning_rate=0.01, warm_start=False, batch_size=128):
        super(ElasticNet, self).__init__(loss_name="mse",
                                         l1_penalty=l1_penalty,
                                         l2_penalty=l2_penalty,
                                         fit_intercept=fit_intercept,
                                         n_iter=n_iter,
                                         verbose=verbose,
                                         random_state=random_state,
                                         learning_rate=learning_rate,
                                         warm_start=warm_start,
                                         batch_size=batch_size)


class Lasso(ElasticNet):
    def __init__(self, fit_intercept=True, l1_penalty=.0001, n_iter=5,
                 verbose=0, random_state=None, learning_rate=0.01,
                 warm_start=False, batch_size=128):
        super(Lasso, self).__init__(l2_penalty=0.,
                                    fit_intercept=fit_intercept,
                                    l1_penalty=l1_penalty,
                                    n_iter=n_iter,
                                    verbose=verbose,
                                    random_state=random_state,
                                    learning_rate=learning_rate,
                                    warm_start=warm_start,
                                    batch_size=batch_size)


class Ridge(ElasticNet):
    def __init__(self, fit_intercept=True, l2_penalty=.0001, n_iter=5,
                 verbose=0, random_state=None, learning_rate=0.01,
                 warm_start=False, batch_size=128):
        super(Ridge, self).__init__(l1_penalty=0.,
                                    fit_intercept=fit_intercept,
                                    l2_penalty=l2_penalty,
                                    n_iter=n_iter,
                                    verbose=verbose,
                                    random_state=random_state,
                                    learning_rate=learning_rate,
                                    warm_start=warm_start,
                                    batch_size=batch_size)


class LinearRegression(Ridge):
    def __init__(self, fit_intercept=True, n_iter=5,
                 verbose=0, random_state=None, learning_rate=0.01,
                 warm_start=False, batch_size=128):
        super(LinearRegression, self).__init__(l2_penalty=0.,
                                               fit_intercept=fit_intercept,
                                               n_iter=n_iter,
                                               verbose=verbose,
                                               random_state=random_state,
                                               learning_rate=learning_rate,
                                               warm_start=warm_start,
                                               batch_size=batch_size)


class L1Regression(SGDRegressor):
    def __init__(self, fit_intercept=True, l1_penalty=.0001, l2_penalty=0.0001,
                 n_iter=5, verbose=0, random_state=None,
                 learning_rate=0.01, warm_start=False, batch_size=128):
        super(ElasticNet, self).__init__(loss_name="mad",
                                         l1_penalty=l1_penalty,
                                         l2_penalty=l2_penalty,
                                         fit_intercept=fit_intercept,
                                         n_iter=n_iter,
                                         verbose=verbose,
                                         random_state=random_state,
                                         learning_rate=learning_rate,
                                         warm_start=warm_start,
                                         batch_size=batch_size)


class LogisticRegression(SGDClassifier):
    def __init__(self, fit_intercept=True, l1_penalty=.0001, l2_penalty=0.0001,
                 n_iter=5, verbose=0, random_state=None,
                 learning_rate=0.01, warm_start=False, batch_size=128):
        super(LogisticRegression, self).__init__(loss_name="log",
                                                 l1_penalty=l1_penalty,
                                                 l2_penalty=l2_penalty,
                                                 fit_intercept=fit_intercept,
                                                 n_iter=n_iter,
                                                 verbose=verbose,
                                                 random_state=random_state,
                                                 learning_rate=learning_rate,
                                                 warm_start=warm_start,
                                                 batch_size=batch_size)


class LinearSVC(SGDClassifier):
    def __init__(self, fit_intercept=True, l1_penalty=.0001, l2_penalty=0.0001,
                 n_iter=5, verbose=0, random_state=None,
                 learning_rate=0.01, warm_start=False, batch_size=128):
        super(LinearSVC, self).__init__(loss_name="hinge",
                                        l1_penalty=l1_penalty,
                                        l2_penalty=l2_penalty,
                                        fit_intercept=fit_intercept,
                                        n_iter=n_iter,
                                        verbose=verbose,
                                        random_state=random_state,
                                        learning_rate=learning_rate,
                                        warm_start=warm_start,
                                        batch_size=batch_size)
