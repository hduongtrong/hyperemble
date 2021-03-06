from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from hyperemble.utils import corr, table_int, table_obj


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
    assert type(res) is dict
    assert len(res) == 3
    assert res == {1: 4, 2: 3, 3: 1}


def test_table_obj():
    x = np.array(["a", "a", "b", "c"])
    res = table_obj(x)
    assert type(res) == dict
    assert len(res) == 3
    assert res == {"a": 2, "b": 1, "c": 1}
