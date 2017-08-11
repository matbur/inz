from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .layers import Layer
from .logger import create_logger
from .utils import it, split

logger = create_logger(__name__, con_level='DEBUG', filename=Path(__file__).with_suffix('.log'))


def loss(a, b) -> np.ndarray:
    return np.sum((a - b) ** 2) / 2


class DNN:
    def __init__(self, network: Layer):
        self.network = network

        layer = network
        while layer.previous is not None:
            layer = layer.previous
        self.input = layer

        self._learn_coef = .2
        self.errors = []

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
            self.errors.append(mean)
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

    def plot_error(self):
        plt.grid()
        y = self.errors
        x = range(len(y))
        plt.plot(x, y)
        plt.scatter(x, y)
        plt.show()

    def show(self, param):
        it(self.network, param)
