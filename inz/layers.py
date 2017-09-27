from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import activation as act
from .logger import create_logger

logger = create_logger(
    __name__,
    # con_level='DEBUG',
    filename=Path(__file__).with_suffix('.log'),
)


def gen_weights():
    yield np.array([[-0.54316974, 3.18716818, -3.45807859],
                    [-0.05402973, -6.34248843, -6.58317728],
                    [0.01800948, 6.57719873, 6.43227908]])

    yield np.array([[-3.45402763, 3.41450737],
                    [-0.49709841, 0.62235418],
                    [7.63121836, -7.63994564],
                    [-7.8834429, 7.88701914]])


weights = gen_weights()


class Layer:
    id = 0

    def __init__(self, shape, activation='sigmoid'):
        self.learning_rate = .2
        self.shape = shape
        self.n_inputs = shape[0]
        self.n_outputs = shape[1]

        self.activation_name = activation
        self.activation = self.parse_activation()

        self.tab: np.ndarray = None
        if shape[0] is not None:
            self.tab = np.random.random((shape[0] + 1, shape[1]))  # - .5
            # self.tab = next(weights)

        self.previous: Layer = None
        self.next: Layer = None

        self.is_first = False
        self.is_last = False

        self.y: np.ndarray = None
        self.z: np.ndarray = None
        self.delta: np.ndarray = None
        self.gradient: np.ndarray = None

        self.id = Layer.id
        Layer.id += 1

    @property
    def W(self):
        if self.tab is None:
            return None
        return self.tab[1:]

    @property
    def b(self):
        if self.tab is None:
            return None
        return self.tab[:1]

    def parse_activation(self):
        return getattr(act, self.activation_name)

    @staticmethod
    def _add_bias(arr: np.ndarray):
        return np.c_[np.ones(arr.shape[0]), arr]

    @staticmethod
    def _add_bias2(arr: np.ndarray):
        return np.concatenate([[1], arr])

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        # I assume that shape of x is [n, 1]
        if self.is_first:
            x = self._add_bias2(x)
            self.y = x
            return self.next.feedforward(x)

        logger.debug('Layer {.id}: got input\n{!r}'.format(self, x))
        z = x @ self.tab
        y = self.activation(z)
        logger.debug('Layer {.id}: returns\n{!r}'.format(self, y))

        if not self.is_last:
            y = self._add_bias2(y)
        self.y = y
        self.z = z

        if self.is_last:
            return y
        return self.next.feedforward(y)

    def feedforward_new(self, x: np.ndarray) -> np.ndarray:
        if self.is_first:
            # if x is 1-D vector, add dimension
            if len(x.shape) == 1:
                x = x[np.newaxis]
            x = self._add_bias(x)
            self.y_new = x
            return self.next.feedforward_new(x)

        logger.debug('Layer {.id}: got input\n{!r}'.format(self, x))
        z = x @ self.tab
        y = self.activation(z)
        logger.debug('Layer {.id}: returns\n{!r}'.format(self, y))

        if not self.is_last:
            y = self._add_bias(y)
        self.y_new = y
        self.z_new = z

        # assert np.array_equal(self.y, self.y2[0])
        # assert np.array_equal(self.z, self.z2[0])

        if self.is_last:
            return y
        return self.next.feedforward_new(y)

    def calc_delta(self, d: np.ndarray = None):
        if self.is_first:
            return

        if self.is_last:
            delta = (self.y - d) * self.activation(self.z, True)
            self.delta = delta
            return self.previous.calc_delta()

        delta = (self.next.delta @ self.next.tab.T)[1:] * self.activation(self.z, True)
        self.delta = delta

        self.previous.calc_delta()

    def calc_delta_new(self, d: np.ndarray = None):
        if self.is_first:
            return

        if self.is_last:
            d = d[None]
            delta = (self.y_new - d) * self.activation(self.z_new, True)
            self.delta_new = delta
            return self.previous.calc_delta_new()

        delta = (self.next.delta @ self.next.tab.T)[1:] * self.activation(self.z_new, True)
        self.delta_new = delta

        assert np.array_equal(self.delta, self.delta_new[0])

        self.previous.calc_delta_new()

    def calc_gradient(self):
        if self.is_first:
            return

        self.gradient = self.previous.y[None].T @ self.delta[None]

        self.previous.calc_gradient()

    def calc_gradient_new(self):
        if self.is_first:
            return

        gradient = self.previous.y_new.T @ self.delta_new
        self.gradient_new = gradient

        assert np.array_equal(self.gradient, self.gradient_new)

        self.previous.calc_gradient_new()

    def update_weights(self):
        if self.is_first:
            return

        self.tab -= self.gradient * self.learning_rate

        self.previous.update_weights()

    def __repr__(self):
        if None in self.shape:
            return f'Layer {self.id}: shape:{self.shape}'
        # TODO: swap W i b
        return f'Layer {self.id}: tab:{self.tab.shape}\n{self.tab} = tab'


def input_data(shape: Tuple[Optional[int], int]) -> Layer:
    layer = Layer(shape)
    layer.is_first = True
    return layer


def fully_connected(incoming: Layer, n_units: int, activation='relu') -> Layer:
    shape = (incoming.n_outputs, n_units)
    layer = Layer(shape, activation)
    layer.previous = incoming
    layer.is_last = True
    incoming.next = layer
    incoming.is_last = False
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
