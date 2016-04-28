from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


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


def head(df):
    """ Return the head of a pandas dataframe

    Parameters
    ----------
    df: pd.DataFrame, shape (n, p)
        The data frame

    Returns
    -------
    head_df: pd.DataFrame, shape (n, 5)
        The first 5 rows of the dataframe

    Notes
    -----
    This function help transitioning from R

    Examples:
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.arange(20))
    """
    return df.head()


def tail(df):
    """ Return the tail of a pandas dataframe

    Parameters
    ----------
    df: pd.DataFrame, shape (n, p)
        The data frame

    Returns
    -------
    tail_df: pd.DataFrame, shape (n, 5)
        The last 5 rows of the dataframe

    Notes
    -----
    This function help transitioning from R
    """
    return df.tail()


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
