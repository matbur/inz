from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inz.layers import Layer, fully_connected, input_data
from inz.logger import create_logger
from inz.utils import split

logger = create_logger(__name__, con_level='DEBUG', filename=Path(__file__).with_suffix('.log'))


def loss(a, b) -> np.ndarray:
    return np.sum((a - b) ** 2) / 2


ERRORS = {'x': [], 'y': []}


class DNN:
    def __init__(self, network: Layer):
        self.network = network

        layer = network
        while layer.previous is not None:
            layer = layer.previous
        self.input = layer

        self._learn_coef = .2

    def fit(self, X_inputs: np.ndarray, Y_targets: np.ndarray, n_epoch=10, batch_size=64, shuffle=False):

        xlen = len(X_inputs)
        for epoch in range(n_epoch):
            # p = np.random.permutation(xlen)
            p = np.arange(xlen)
            batches_x = split(X_inputs[p], batch_size)
            batches_y = split(Y_targets[p], batch_size)
            err = []
            for batch_x, batch_y in zip(batches_x, batches_y):
                self.input.feedforward(batch_x)
                self.network.calc_delta(batch_y)
                self.network.calc_gradient()
                self.network.update_weights()
                e = loss(self.network.y, batch_y)
                err.append(e)
            mean = np.mean(err)
            ERRORS['x'].append(epoch)
            ERRORS['y'].append(mean)
            logger.info('epoch: {}| error: {:.6f}'.format(epoch, mean))

    def get_weights(self):
        pass

    def load(self, model_file: str):
        pass

    def save(self, model_file: str):
        pass

    def predict(self, X: list):
        return self.input.feedforward(X)

    def predict_label(self, X: list):
        pass

    def set_weights(self, tensor, weights):
        pass


def it(network: Layer, attr, i=0):
    values = getattr(network, attr)
    previous = network.previous

    if previous is None:
        return
    print(attr)
    print('Layer: {}, i: {} {}.shape: {}'.format(network.id, i, attr, values.shape))
    print(values)
    it(previous, attr, i + 1)


def main():
    np.random.seed(42)

    x = np.array(([1, 1], [1, 0], [0, 1], [0, 0]), dtype=float)
    y = np.array(([1, 0], [0, 1], [0, 1], [1, 0]), dtype=float)

    inp = input_data((None, 2))
    fc1 = fully_connected(inp, 3)
    fc2 = fully_connected(fc1, 2)

    model = DNN(fc2)
    model.fit(x, y, n_epoch=3000, batch_size=1)

    # it(model.network, 'W')
    print(inp)
    print(fc1)
    print(fc2)

    plt.grid()
    plt.plot(ERRORS['x'], ERRORS['y'])
    plt.show()


if __name__ == '__main__':
    main()
