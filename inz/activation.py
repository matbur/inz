"""

source: https://en.wikipedia.org/wiki/Activation_function
"""
from typing import Iterable, List

import numpy as np


def identity(x: float, derivative=False) -> float:
    if derivative:
        return 1
    return x


def binary_step(x: float, derivative=False) -> float:
    if derivative:
        if x == 0:
            raise ValueError('?')
        return 0

    return int(x >= 0)


def logistic(x: float, derivative=False) -> float:
    if derivative:
        fx = logistic(x)
        return fx * (1 - fx)
    return 1 / (1 + np.exp(-x))


def tanh(x: float, derivative=False) -> float:
    if derivative:
        return 1 - tanh(x) ** 2
    return np.tanh(x)


def arctan(x: float, derivative=False) -> float:
    if derivative:
        return 1 / (1 + x ** 2)
    return np.arctan(x)


def soft_sign(x: float, derivative=False) -> float:
    if derivative:
        return 1 / (1 + abs(x)) ** 2
    return x / (1 + abs(x))


def relu(x: float, derivative=False) -> float:
    if derivative:
        return int(x >= 0)
    return max(0., x)


def softmax(x: Iterable[float]) -> List[float]:
    s = sum(np.exp(x))
    return [np.exp(i) / s for i in x]
