"""

source: https://en.wikipedia.org/wiki/Activation_function
"""
from typing import Iterable, List

import numpy as np


def identity(x, derivative=False):
    if derivative:
        return 1
    return x


def binary_step(x, derivative=False):
    if derivative:
        if x == 0:
            raise ValueError('?')
        return 0

    return x >= 0


def sigmoid(x, derivative=False):
    if derivative:
        return np.exp(x) / (1 + np.exp(x)) ** 2
    return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 1 - tanh(x) ** 2
    return np.tanh(x)


def arctan(x, derivative=False):
    if derivative:
        return 1 / (1 + x ** 2)
    return np.arctan(x)


def soft_sign(x, derivative=False):
    if derivative:
        return 1 / (1 + abs(x)) ** 2
    return x / (1 + abs(x))


def relu(x, derivative=False):
    if derivative:
        return int(x >= 0)
    return max(0., x)


def softmax(x: Iterable[float]) -> List[float]:
    s = sum(np.exp(x))
    return [np.exp(i) / s for i in x]
