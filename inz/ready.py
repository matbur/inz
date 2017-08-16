import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tflearn import DNN, input_data, regression, fully_connected


def use_sklearn(x_train, y_train, x_test, y_test):
    model = MLPClassifier()
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def use_tflearn(x_train, y_train, x_test, y_test):
    input_ = input_data(shape=[None, 2])
    fc1 = fully_connected(input_, 3, activation='sigmoid')
    fc2 = fully_connected(fc1, 2, activation='sigmoid')
    regression_ = regression(fc2, optimizer='sgd')
    model = DNN(regression_)
    model.fit(x_train, y_train, validation_set=(x_test, y_test), n_epoch=10 ** 5)
    return model.predict_label(x_test)


def main():
    file = Path('./data.json').read_text()
    data = np.array(json.loads(file))

    train, test = train_test_split(data)
    x_train = train[:, :-1].reshape(-1, 1)
    y_train = train[:, -1].reshape(-1, 1)
    x_test = test[:, :-1].reshape(-1, 1)
    y_test = test[:, -1].reshape(-1, 1)

    np.random.seed(42)

    x = np.array([
        [1, 1], [1, 0], [0, 1], [0, 0],
        [1, 1], [1, 0], [0, 1], [0, 0],
        [1, 1], [1, 0], [0, 1], [0, 0],
        [1, 1], [1, 0], [0, 1], [0, 0],
    ], dtype=float)
    y = np.array([
        [1, 0], [0, 1], [0, 1], [1, 0],
        [1, 0], [0, 1], [0, 1], [1, 0],
        [1, 0], [0, 1], [0, 1], [1, 0],
        [1, 0], [0, 1], [0, 1], [1, 0],
    ], dtype=float)

    print(use_sklearn(x, y, x, y))
    # print(use_tflearn(x, y, x, y))


if __name__ == '__main__':
    main()
