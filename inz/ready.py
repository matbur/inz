import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from tflearn import DNN, input_data, regression, single_unit


def use_sklearn(x_train, y_train, x_test, y_test):
    model = MLPRegressor()
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def use_tflearn(x_train, y_train, x_test, y_test):
    input_ = input_data(shape=[None])
    linear = single_unit(input_)
    regression_ = regression(linear)
    model = DNN(regression_)
    model.fit(x_train, y_train)
    return model.predict_label(x_test)


def main():
    file = Path('./data.json').read_text()
    data = np.array(json.loads(file))

    train, test = train_test_split(data)
    x_train = train[:, :-1].reshape(-1, 1)
    y_train = train[:, -1].reshape(-1, 1)
    x_test = test[:, :-1].reshape(-1, 1)
    y_test = test[:, -1].reshape(-1, 1)

    print(use_sklearn(x_train, y_train, x_test, y_test))


if __name__ == '__main__':
    main()
