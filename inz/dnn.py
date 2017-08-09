import numpy as np

from inz import activation as act
from inz.layers import Layer, fully_connected, input_data


class DNN:
    def __init__(self, network: Layer):
        self.network = network

        layer = network
        while layer.previous is not None:
            layer = layer.previous
        self.input = layer

    def fit(self, X_inputs: list, Y_targets: list, n_epoch=10, batch_size=64, shuffle=False):
        pass

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


if __name__ == '__main__':
    x = np.array(([1, 1], [1, 0], [0, 1], [0, 0]), dtype=float)
    y = np.array(([1, 0], [0, 1], [0, 1], [1, 0]), dtype=float)

    inp = input_data((None, 2))
    fc1 = fully_connected(inp, 3)
    fc2 = fully_connected(fc1, 2, activation=act.softmax)

    model = DNN(fc2)

    np_round = np.round(model.predict(x), 3)

    for i, j, k in zip(x, np_round, y):
        print(i, j, k)
