import numpy as np
import pytest

from inz import split


def test_split_list_int():
    ints = list(range(7))
    want = [[0, 1, 2], [3, 4, 5]]
    get = list(split(ints, 3))
    assert get == want


def test_split_list_str():
    strings = list(map(str, range(6)))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2))
    assert get == want


def test_split_ndarray_int():
    array = np.arange(10, dtype=int).reshape(-1, 2)
    want = [np.array([[0, 1], [2, 3]]),
            np.array([[4, 5], [6, 7]])]
    get = list(split(array, 2))
    assert len(get) == len(want)
    assert all([(i == j).all() for i, j in zip(get, want)])


def test_split_generator_str():
    strings = map(str, range(6))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2))
    assert get == want


if __name__ == '__main__':
    pytest.main()
