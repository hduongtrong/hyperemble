from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from keras.datasets import mnist
from hyperemble.neural_net import VanillaNeuralNet


def test_vanilla_neural_net():
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
    res = clf.score(X_test, y_test)
    assert res > 0.92
