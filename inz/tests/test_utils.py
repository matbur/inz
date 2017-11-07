import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, chi2 as sk_chi2

from inz.utils import chi2, select_k_best, split, train_test_split


def test_split_list_int():
    ints = list(range(7))
    want = [[0, 1, 2], [3, 4, 5], [6]]
    get = list(split(ints, 3))
    assert len(get) == len(want)
    assert get == want


def test_split_int():
    ints = range(7)
    want = [[0, 1, 2], [3, 4, 5], [6]]
    get = list(split(ints, 3))
    assert len(get) == len(want)
    assert get == want


def test_split_list_int_greater_width():
    ints = list(range(3))
    want = [[0, 1, 2]]
    get = list(split(ints, 4))
    assert len(get) == len(want)
    assert get == want


def test_split_list_str():
    strings = list(map(str, range(6)))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2))
    assert len(get) == len(want)
    assert get == want


def test_str():
    string = ''.join(map(str, range(6)))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(string, 2))
    assert len(get) == len(want)
    assert get == want


def test_split_ndarray_int():
    array = np.arange(10, dtype=int).reshape(-1, 2)
    want = [np.array([[0, 1], [2, 3]]),
            np.array([[4, 5], [6, 7]]),
            np.array([[8, 9]])]
    get = list(split(array, 2))
    assert len(get) == len(want)
    for i, j in zip(get, want):
        assert type(i) == type(j)
        assert np.array_equal(i, j)


def test_split_generator_str():
    strings = map(str, range(6))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2))
    assert len(get) == len(want)
    assert get == want


def test_split_list_int_not_allow():
    ints = list(range(7))
    want = [[0, 1, 2], [3, 4, 5]]
    get = list(split(ints, 3, False))
    assert len(get) == len(want)
    assert get == want


def test_split_list_int_greater_width_not_allow():
    ints = list(range(3))
    want = []
    get = list(split(ints, 4, False))
    assert len(get) == len(want)
    assert get == want


def test_split_list_str_not_allow():
    strings = list(map(str, range(6)))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2, False))
    assert len(get) == len(want)
    assert get == want


def test_split_ndarray_int_not_allow():
    array = np.arange(10, dtype=int).reshape(-1, 2)
    want = [np.array([[0, 1], [2, 3]]),
            np.array([[4, 5], [6, 7]])]
    get = list(split(array, 2, False))
    assert len(get) == len(want)
    for i, j in zip(get, want):
        assert type(i) == type(j)
        assert np.array_equal(i, j)


def test_split_generator_str_not_allow():
    strings = map(str, range(6))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2, False))
    assert len(get) == len(want)
    assert get == want


@pytest.fixture
def data():
    X = pd.read_csv('../../data/data.csv')
    y = X.pop('Choroba')
    return X.values, y.values


def test_chi2(data):
    X, y = data
    sk_val, _ = sk_chi2(X, y)
    my_val = chi2(X, y)

    np.testing.assert_equal(sk_val, my_val)


def test_select_k_best(data):
    X, y = data
    for i in range(1, 31):
        sk_sup1 = SelectKBest(sk_chi2, i).fit(X, y).get_support()
        sk_sup2 = SelectKBest(sk_chi2, i).fit(X, y).get_support(True)

        my_sup1 = select_k_best(X, y, k=i)
        my_sup2 = select_k_best(X, y, k=i, indices=True)

        np.testing.assert_equal(sk_sup1, my_sup1, str(i))
        np.testing.assert_equal(sk_sup2, sorted(my_sup2), str(i))


def test_train_test_split():
    x = np.arange(10)
    get = train_test_split(x, shuffle=False)
    want = [np.arange(7), np.arange(7, 10)]
    for i in zip(get, want):
        np.testing.assert_equal(*i)


def test_train_test_split5():
    x = np.arange(10)
    get = train_test_split(x, test_size=.5, shuffle=False)
    want = [np.arange(5), np.arange(5, 10)]
    for i in zip(get, want):
        np.testing.assert_equal(*i)


if __name__ == '__main__':
    pytest.main()
