from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import pandas as pd
from hyperemble.utils import corr, table_int, table_obj, head, tail


def test_corr():
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    c = np.array([2, 3, 5])
    d = corr(a, b)
    e = corr(a, c)
    assert type(d) is np.float64
    assert type(e) is np.float64
    assert np.isclose(d, 1.)
    assert np.isclose(e, 0.98198050606196585)


def test_table_int():
    x = np.array([1, 1, 1, 1, 2, 2, 2, 3])
    res = table_int(x)
    assert type(res) is list
    assert len(res) == 3
    assert res == [(1, 4), (2, 3), (3, 1)]


def test_table_obj():
    x = np.array(["a", "a", "b", "c"])
    res = table_obj(x)
    assert type(res) == dict
    assert len(res) == 3
    assert res == {"a": 2, "b": 1, "c": 1}


def test_head():
    mat = np.arange(20)
    mat = mat.reshape(10, 2)
    df = pd.DataFrame(mat)
    df_head = head(df)
    assert type(df_head) is pd.DataFrame
    assert df_head.shape == (5, 2)
    assert df_head.iloc[0, 0] == 0


def test_tail():
    mat = np.arange(20)
    mat = mat.reshape(10, 2)
    df = pd.DataFrame(mat)
    df_head = tail(df)
    assert type(df_head) is pd.DataFrame
    assert df_head.shape == (5, 2)
    assert df_head.iloc[0, 0] == 10
