import numpy as np
import pytest

from inz import split


def test_split_list_int():
    ints = list(range(7))
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


def test_split_ndarray_int():
    array = np.arange(10, dtype=int).reshape(-1, 2)
    want = [np.array([[0, 1], [2, 3]]),
            np.array([[4, 5], [6, 7]]),
            np.array([[8, 9]])]
    get = list(split(array, 2))
    assert len(get) == len(want)
    assert all([np.array_equal(i, j) for i, j in zip(get, want)])


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
    assert all([np.array_equal(i, j) for i, j in zip(get, want)])


def test_split_generator_str_not_allow():
    strings = map(str, range(6))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2, False))
    assert len(get) == len(want)
    assert get == want

if __name__ == '__main__':
    pytest.main()
