from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import activation as act
from .logger import create_logger

logger = create_logger(
    __name__,
    con_level='DEBUG',
    filename=Path(__file__).with_suffix('.log'),
)


def gen_weights():
    yield np.array([[-0.05402973, -6.34248843, -6.58317728],
                    [0.01800948, 6.57719873, 6.43227908]])
    yield np.array([[-0.54316974, 3.18716818, -3.45807859]])

    yield np.array([[-0.49709841, 0.62235418],
                    [7.63121836, -7.63994564],
                    [-7.8834429, 7.88701914]])
    yield np.array([[-3.45402763, 3.41450737]])


weights = gen_weights()


class Layer:
    id = 0

    def __init__(self, shape, activation=act.sigmoid):
        self.learn_coef = .2
        self.shape = shape
        self.n_inputs = shape[0]
        self.n_outputs = shape[1]

        self.activation = activation

        self.W = None
        self.b = None
        if shape[0] is not None:
            # self.W = np.random.random(shape) - .5
            # self.b = np.random.random((1, shape[1])) - .5

            self.b = np.random.random((1, shape[1]))
            self.W = np.random.random(shape)
            # self.W = next(weights)
            # self.b = next(weights)

        self.previous: Layer = None
        self.next: Layer = None

        self.y = None
        self.z = None
        self.delta = None
        self.gradient = None

        self.id = Layer.id
        Layer.id += 1

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        is_first = self.previous is None
        is_last = self.next is None

        # logger.debug('Layer {.id}: got input\n{!r}'.format(self, x))
        if is_first:
            # logger.debug("Layer {.id}: it's first, so pass".format(self))
            self.y = x
            return self.next.feedforward(x)

        # logger.debug('x.shape: {}'.format(x.shape))
        # logger.debug('W.shape: {}'.format(self.W.shape))
        # logger.debug('b.shape: {}'.format(self.b.shape))
        # logger.debug('predicted y.shape: {}'.format((x.shape[0], self.W.shape[1])))

        m = x @ self.W
        # logger.debug('Layer {.id}: multiplied input by weights\n{!r}'.format(self, m))
        a = m + self.b
        # logger.debug('Layer {.id}: added bias\n{!r}'.format(self, a))
        y = self.activation(a)
        # logger.debug('Layer {.id}: returns\n{!r}'.format(self, y))

        self.y = y
        self.z = a

        return y if is_last else self.next.feedforward(y)

    def calc_delta(self, d=None):
        is_first = self.previous is None
        is_last = self.next is None

        if is_first:
            return

        if is_last:
            self.delta = (self.y - d) * self.activation(self.z, True)
            return self.previous.calc_delta()

        self.delta = np.dot(self.next.delta, self.next.W.T) * self.activation(self.z, True)

        self.previous.calc_delta()

    def calc_gradient(self):
        is_first = self.previous is None
        is_last = self.next is None

        if is_first:
            return

        y = np.r_[[1], self.previous.y[0]][np.newaxis].T
        delta = self.delta  # .mean(axis=0)[np.newaxis]
        dot = np.dot(y, delta)
        self.gradient = dot

        self.previous.calc_gradient()

    def update_weights(self):
        is_first = self.previous is None
        is_last = self.next is None

        if is_first:
            return

        adjustment_W = self.gradient[1:]
        adjustment_b = self.gradient[:1]

        self.W -= adjustment_W * self.learn_coef
        self.b -= adjustment_b * self.learn_coef

        self.previous.update_weights()

    def __repr__(self):
        if None in self.shape:
            return f'Layer {self.id}: shape:{self.shape}'
        # TODO: swap W i b
        return f'Layer {self.id}: W:{self.W.shape} b:{self.b.shape}\n{self.b} = b\n{self.W} = W'


def input_data(shape: Tuple[Optional[int], int]) -> Layer:
    return Layer(shape)


def fully_connected(incoming: Layer, n_units: int, activation=act.sigmoid) -> Layer:
    shape = (incoming.n_outputs, n_units)
    layer = Layer(shape, activation)
    layer.previous = incoming
    incoming.next = layer
    return layer


def dropout(incoming: Layer, keep_prob=.8) -> Layer:
    pass


def regression(incoming, optimizer='adam'):
    def loss(a, b) -> np.ndarray:
        return np.sum((a - b) ** 2) / 2

    shape = (None, None)
    layer = Layer(shape)
    layer.previous = incoming
    return layer
