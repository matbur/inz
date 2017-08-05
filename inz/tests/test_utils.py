import pytest

from inz import split


def test_split_int():
    ints = list(range(7))
    want = [[0, 1, 2], [3, 4, 5]]
    get = list(split(ints, 3))
    assert get == want


def test_split_str():
    strings = list(map(str, range(6)))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2))
    assert get == want


def test_split_generator():
    strings = map(str, range(6))
    want = [['0', '1'], ['2', '3'], ['4', '5']]
    get = list(split(strings, 2))
    assert get == want


if __name__ == '__main__':
    pytest.main()
