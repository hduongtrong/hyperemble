from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from keras.datasets import mnist
from hyperemble.linear_model import (SGDRegressor, SGDClassifier,
                                     SoftmaxRegression)


def test_sgd_regressor_noise():
    n, p = 10000, 10
    X = np.random.randn(n, p)
    X = np.array(X, dtype=np.float32)
    beta = np.random.randint(low=0, high=3, size=p)
    signal = X.dot(beta) + 1
    y = signal + np.std(signal) * np.random.randn(n)
    clf = SGDRegressor(loss_name="mad", n_iter=20, l1_penalty=.0001,
                       l2_penalty=.0001, verbose=0)
    clf.fit(X, y)
    yhat = clf.predict(X)
    res = r2_score(y, yhat)
    assert res > 0.4
    assert res < 0.6


def test_sgd_regressor_no_noise():
    n, p = 10000, 10
    X = np.random.randn(n, p)
    X = np.array(X, dtype=np.float32)
    beta = np.random.randint(low=0, high=3, size=p)
    signal = X.dot(beta) + 1
    y = signal
    clf = SGDRegressor(loss_name="mad", n_iter=20, l1_penalty=.0001,
                       l2_penalty=.0001, verbose=0)
    clf.fit(X, y)
    yhat = clf.predict(X)
    res = r2_score(y, yhat)
    assert res > 0.9


def test_sgd_classifier_no_noise():
    n, p = 10000, 10
    X = np.random.randn(n, p)
    X = np.array(X, dtype=np.float32)
    beta = np.random.randint(low=0, high=3, size=p)
    signal = X.dot(beta)
    y = np.ones(len(signal), dtype=np.int64)
    y[signal < 0] = -1
    clf = SGDClassifier(loss_name="hinge", n_iter=20, l1_penalty=.0001,
                        l2_penalty=.0001, verbose=0,
                        learning_rate=0.01, fit_intercept=False)
    clf.fit(X, y)
    yhat = clf.predict(X)
    res = accuracy_score(y, yhat)
    assert res > 0.95


def test_softmax_regression():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    clf = SoftmaxRegression(verbose=0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    assert acc > 0.85
    assert acc < 0.95
