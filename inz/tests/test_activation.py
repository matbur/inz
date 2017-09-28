from inz.activation import relu

import numpy as np


def test_relu_matrix():
    arr = np.array([[1, -1, .5, -.5, 1e-6, -1e6]])
    want = np.array([[1, 0, .5, 0, 1e-6, 0]])
    get = relu(arr)
    assert np.array_equal(want, get)


def test_relu_vector():
    arr = np.array([1, -1, .5, -.5, 1e-6, -1e6])
    want = np.array([1, 0, .5, 0, 1e-6, 0])
    get = relu(arr)
    assert np.array_equal(want, get)


def test_relu_der_matrix():
    arr = np.array([[1, -1, .5, -.5, 1e-6, -1e6]])
    want = np.array([[1., 0., 1., 0., 1., 0.]])
    get = relu(arr, True)
    assert np.array_equal(want, get)


def test_relu_der_vector():
    arr = np.array([1, -1, .5, -.5, 1e-6, -1e6])
    want = np.array([1., 0., 1., 0., 1., 0.])
    get = relu(arr, True)
    assert np.array_equal(want, get)
