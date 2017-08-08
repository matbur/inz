import logging
from typing import Optional, Tuple

import numpy as np

from inz import activation as act


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

    def __init__(self, shape, activation=act.binary_step):
        self.shape = shape
        self.n_inputs = shape[0]
        self.n_outputs = shape[1]

        self.activation = activation

        self.W = None
        self.b = None
        if shape[0] is not None:
            # self.W = np.random.random(shape) - .5
            # self.b = np.random.random((1, shape[1])) - .5
            self.W = next(weights)
            self.b = next(weights)

        self.previous: Layer = None
        self.next: Layer = None

        self.id = Layer.id
        Layer.id += 1

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        is_first = self.previous is None
        is_last = self.next is None

        logging.warning('Layer {.id}: got input np.{!r}'.format(self, x))
        if is_first:
            return self.next.feedforward(x)

        logging.warning('x.shape: {}'.format(x.shape))
        logging.warning('W.shape: {}'.format(self.W.shape))
        logging.warning('b.shape: {}'.format(self.b.shape))
        logging.warning('predicted y.shape: {}'.format((x.shape[0], self.W.shape[1])))

        m = x @ self.W
        logging.warning('Layer {.id}: multiplied np.{!r}'.format(self, m))
        m += self.b
        logging.warning('Layer {.id}: added np.{!r}'.format(self, m))
        y = self.activation(m)
        logging.warning('Layer {.id}: returns np.{!r}'.format(self, y))

        return y if is_last else self.next.feedforward(y)

    def backpropagate(self):
        pass

    def __repr__(self):
        if self.id == 0:
            return f'Layer {self.id}: shape:{self.shape}'
        return f'Layer {self.id}: W:{self.W.shape} b:{self.b.shape}\n{self.W} = W\n{self.b} = b'


def input_data(shape: Tuple[Optional[int], int]) -> Layer:
    return Layer(shape)


def fully_connected(incoming: Layer, n_units: int, activation=act.binary_step) -> Layer:
    shape = (incoming.n_outputs, n_units)
    layer = Layer(shape, activation)
    layer.previous = incoming
    incoming.next = layer
    return layer


def dropout(incoming: Layer, keep_prob=.8) -> Layer:
    pass


def regression(incoming, optimizer='adam'):
    pass


if __name__ == '__main__':
    np.random.seed(42)

    inp = input_data(shape=(None, 2))
    fc1 = fully_connected(inp, 3, activation=act.sigmoid)
    fc2 = fully_connected(fc1, 2, activation=act.sigmoid)

    x = np.array(([1, 1], [1, 0], [0, 1], [0, 0]), dtype=float)
    y = np.array(([1, 0], [0, 1], [0, 1], [1, 0]), dtype=float)

    print(inp)
    print(fc1)
    print(fc2)

    np_round = np.round(inp.feedforward(x), 3)

    for i, j, k in zip(x, np_round, y):
        print(i, j, k)
