from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import collections
import numpy as np
import tensorflow as tf


class PrintMess():
    def __init__(self):
        self.time = time.time()

    def info(self, header=True, **kwargs):
        if not header:
            time_spent = time.time() - self.time
            self.time = time.time()
        else:
            time_spent = 0
        kwargs = collections.OrderedDict(sorted(kwargs.items()))
        headers = ["%8s" % "Seconds"]
        messages = ["%8d" % time_spent]
        for key, value in kwargs.iteritems():
            headers.append("%8s" % key)
            if isinstance(value, int):
                messages.append("%8d" % value)
            else:
                messages.append("%8.4f" % value)
        if header:
            print("|".join(headers))
        else:
            print("|".join(messages))


class ProcessBatch():
    """ Class to divide data (X, y) into batch for stochastic gradient descent
    """
    def __init__(self, x, y):
        self.index = 0
        self.x = x
        self.y = y
        self.num_examples = len(x)
        self.epochs_completed = 0
        assert (len(x)) == len(y)

    def next_batch(self, batch_size):
        if self.index + batch_size < self.num_examples:
            batch_index = np.arange(self.index, self.index + batch_size,
                                    dtype=int)
        else:
            batch_index = np.concatenate([
                    np.arange(self.index, self.num_examples),
                    np.arange(0, batch_size - self.num_examples + self.index)])
            self.epochs_completed += 1
        self.index = (self.index + batch_size) % self.num_examples
        return self.x[batch_index], self.y[batch_index]


def logistic_loss(labels, yhat):
    return tf.reduce_mean(tf.log(1. + tf.exp(-labels * yhat)))


def hinge_loss(labels, yhat):
    return tf.reduce_mean(tf.maximum(0., 1 - labels * yhat))


def exp_loss(labels, yhat):
    return tf.reduce_mean(tf.exp(-labels * yhat))


def squared_hinge_loss(labels, yhat):
    return tf.reduce_mean(tf.maximum(0., tf.square(1 - labels * yhat)))


def mean_squared_error(labels, yhat):
    return tf.reduce_mean(tf.square(labels - yhat))


def mean_absolute_deviation(labels, yhat):
    return tf.reduce_mean(tf.abs(labels - yhat))


def corr(x, y):
    """ Compute the correlation between two arrays

    Parameters
    ----------
    x: ndarray, shape (n, 1)
        The first vector
    y: ndarray, shape (n, 1)
        The second vector of same shape

    Returns
    -------
    corr: float
        The correlation between x and y

    Notes
    -----
    This is a wrapper for np.corrcoef, since np.corrcoef return a 2x2
    matrix, when often we only need the correlation

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> c = np.array([2, 3, 5])
    >>> corr(a, b)
    1.0
    >>> corr(a, c)
    0.98198050606196585
    """
    return np.corrcoef(x, y)[0, 1]


def table_int(x):
    """ Return the count of each element in x

    Parameters
    ----------
    x: np.ndarray, dtype np.int, shape (n, )
        Array to count elements

    Returns
    -------
    y: list
        A list of tuple (a, b) where a is an element, b is the count

    Examples
    --------
    >>> a = np.array([1,1,1,2,2,3])
    >>> table_int(a)
    {1: 3, 2: 2, 3: 1}
    """
    if type(x) is list:
        x = np.array(x)
    assert np.issubdtype(x.dtype, np.integer)
    elements, counts = np.unique(x, return_counts=True)
    return dict(zip(elements, counts))


def table_obj(x):
    """ Return the count of each element in x

    Parameters
    ----------
    x: np.ndarray, shpe (n, 1)
        Array to count elements

    Returns
    -------
    y: dict
        A dictionary contain the element and its count in array x

    Examples
    --------
    >>> x = np.array(["a", "a", "b", "c"])
    >>> res = table_obj(x)
    >>> res['a'], res['b'], res['c']
    (2, 1, 2)
    """
    if type(x) is list:
        x = np.array(x)
    elements = np.unique(x)
    counts = {}
    for element in elements:
        counts[element] = np.sum(x == element)
    return counts

if __name__ == "__main__":
    import doctest
    doctest.testmod()
