import json
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np

from .layers import Layer
from .logger import create_logger
from .utils import iter_layers, split

logger = create_logger(__name__, con_level='DEBUG', filename=Path(__file__).with_suffix('.log'))


def loss(a, b) -> np.ndarray:
    return np.sum((a - b) ** 2) / 2


class Model:
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
        step = 0
        for epoch in range(n_epoch):
            # p = np.random.permutation(xlen)
            p = np.arange(xlen)
            batches_x = split(X_inputs[p], batch_size)
            batches_y = split(Y_targets[p], batch_size)
            err = []
            for i, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                step += 1
                t0 = time()
                for x, y in zip(batch_x, batch_y):
                    self.input.feedforward(x)
                    self.network.calc_delta(y)
                    self.network.calc_gradient()
                    self.network.update_weights()

                e = loss(self.predict(batch_x), batch_y)
                err.append(e)
                t = time() - t0
                print(f'Training Step: {step} | total loss: {e} | time: {t}s')
                print(f'epoch: {epoch} | acc: 0.5191 -- iter: {i * batch_size}/{xlen}')

            mean = np.mean(err)
            self.errors.append(mean)
            # logger.info('epoch: {}| error: {:.6f}'.format(epoch, mean))

    def get_weights(self):
        pass

    def load(self, model_file: str):
        data = json.loads(Path(model_file).read_text())
        it = iter(data)
        layer = self.network
        while layer.previous is not None:
            tab = next(it)
            layer.tab = np.array(tab)
            layer = layer.previous

    def save(self, model_file: str):
        data = []
        layer = self.network
        while layer.previous is not None:
            data.append(layer.tab.tolist())
            layer = layer.previous

        with open(model_file, 'w') as f:
            json.dump(data, f)

    def predict(self, x: np.ndarray):
        return self.input.feedforward(x)

    def predict_label(self, x: np.ndarray):
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

    def show(self, param, with_values=True):
        if isinstance(param, str):
            param = [param]
        for i in param:
            iter_layers(self.network, i, with_values)
