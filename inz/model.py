import json
from itertools import repeat
from pathlib import Path
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .layers import Layer
from .logger import create_logger
from .schemas import NetworkSchema
from .utils import get_accuracy, get_loss, iter_layers, split

logger = create_logger(
    __name__,
    con_level='DEBUG',
    filename=Path(__file__).with_suffix('.log')
)


class Model:
    def __init__(self, network: Layer):
        self.network = network

        layer = network
        while layer.previous is not None:
            layer = layer.previous
        self.input = layer

        self.errors = []

    def fit(self, X_inputs: np.ndarray, Y_targets: np.ndarray,
            validation_set: Tuple[np.ndarray, np.ndarray] = None,
            learning_rate=None, n_epoch=10, batch_size=64,
            shuffle=False, train_file='train.json'):

        xlen = len(X_inputs)
        step = 0
        order = np.random.permutation(xlen) if shuffle else np.arange(xlen)

        if validation_set is None:
            validation_set = X_inputs, Y_targets

        training = []
        testing = []
        for epoch in range(1, n_epoch + 1):
            batches_x = split(X_inputs[order], batch_size)
            batches_y = split(Y_targets[order], batch_size)
            err = []
            num, den = 0, 0
            for i, (batch_x, batch_y) in enumerate(zip(batches_x, batches_y)):
                step += 1
                t0 = time()
                for x, y in zip(batch_x, batch_y):
                    self.input.feedforward(x)
                    self.network.calc_delta(y)
                    self.network.calc_gradient()
                    self.network.update_weights()

                predict = self.predict(batch_x)
                e = get_loss(predict, batch_y)
                err.append(e)

                l = len(batch_x)
                num += l - np.count_nonzero(predict.argmax(axis=1) - batch_y.argmax(axis=1))
                den += l
                acc = num / den
                iteration = i * batch_size

                t = time() - t0
                print(f'Training Step: {step:<4} | total loss: {e:.5f} | time: {t:.3f}s')
                print(f'         epoch: {epoch:0>3} | acc: {acc:.4f} -- iter: {iteration:0>3}/{xlen}')
                training.append({
                    'step': step,
                    'total_loss': e,
                    'time': t,
                    'epoch': epoch,
                    'accuracy': acc,
                    'iteration': iteration,
                    'len': xlen,
                })

            test_predict = self.predict(validation_set[0])
            test_y = validation_set[1]
            test_acc = get_accuracy(test_predict, test_y)
            test_loss = get_loss(test_predict, test_y)

            print('--')
            print(f'End of epoch: {epoch:0>3}   | val_loss: {test_loss:.5f} | val_acc: {test_acc:.4f}')
            print('--')
            testing.append({
                'test_loss': test_loss,
                'epoch': epoch,
                'test_accuracy': test_acc,
            })
            mean = np.mean(err)
            self.errors.append(mean)
        data = json.dumps({'training': training, 'testing': testing})
        Path(train_file).write_text(data)

    @staticmethod
    def _parse_learning_rate(lr, n_epoch=None):
        if isinstance(lr, float):
            return repeat(lr)
        if isinstance(lr, (list, tuple)):
            assert n_epoch

            lr1, lr2 = lr

            def f():
                x = 0
                while 1:
                    yield x * (lr2 - lr1) / (n_epoch - 1) + lr1
                    x += 1

            return f()

    def _apply_lr(self, lr):
        layer = self.network
        while layer.previous is not None:
            layer.learning_rate = lr
            layer = layer.previous

    def get_weights(self) -> NetworkSchema:
        data = []
        layer = self.network
        while layer.previous is not None:
            tab = layer.tab.tolist()
            data.append(tab)
            layer = layer.previous
        return data

    def load(self, model_file: str):
        data_text = Path(model_file).read_text()
        data = json.loads(data_text)
        self.set_weights(data)

    def save(self, model_file: str):
        data = self.get_weights()

        with open(model_file, 'w') as f:
            json.dump(data, f)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.input.feedforward(x)

    def predict_label(self, x: np.ndarray):
        pass

    def set_weights(self, tensor: NetworkSchema):
        self.network.load(tensor[::-1])

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
