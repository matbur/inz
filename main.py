import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from inz import Model, fully_connected, input_data
from inz.utils import select_k_best, vector2onehot


def xor_problem():
    np.random.seed(1)
    x = np.array([
        [1, 1], [1, 0], [0, 1], [0, 0],
    ], dtype=float)
    y = np.array([
        [1, 0], [0, 1], [0, 1], [1, 0],
    ], dtype=float)

    net = input_data((None, 2))
    net = fully_connected(net, 3, activation='tanh')
    net = fully_connected(net, 2, activation='tanh')

    model = Model(net)
    model.fit(x, y, n_epoch=200)
    model.save('model.json')
    model.load('model.json')

    model.show('tab')

    # print(model.predict(x))
    for i in x:
        print(model.predict(i))
    print(y)

    model.plot_error()


def get_data(num_features=20):
    X = pd.read_csv('./data/data.csv')
    y = X.pop('Choroba')

    sup = select_k_best(X, y, k=num_features)

    X = X.drop(X.columns[~sup], axis=1)

    X['Choroba'] = y

    return X.values


def get_accuracy(pred, y):
    axis_ = np.argmax(pred, axis=1) - np.argmax(y, axis=1)
    return 1 - np.count_nonzero(axis_) / len(y)


def main():
    np.random.seed(42)

    data = get_data(20)

    train, test = train_test_split(data)
    print(train.shape, test.shape)

    x_train = train[:, :-1]
    y_train = train[:, -1] - 1
    y_train = vector2onehot(y_train)
    x_test = test[:, :-1]
    y_test = test[:, -1] - 1
    y_test = vector2onehot(y_test)

    net = input_data(shape=(None, x_train.shape[1]))
    net = fully_connected(net, 24, activation='sigmoid')
    net = fully_connected(net, 16, activation='sigmoid')
    net = fully_connected(net, 12, activation='sigmoid')
    net = fully_connected(net, 8, activation='sigmoid')

    model_file = 'model.json'
    model = Model(net)
    model.fit(x_train, y_train,
              validation_set=(x_test, y_test),
              n_epoch=30,
              batch_size=10,
              )
    model.save(model_file)
    model.load(model_file)

    # for i, j in zip(model.predict(x_test), y_test):
    #     print(np.argmax(i), np.argmax(j))

    print(get_accuracy(model.predict(x_test), y_test))

    # model.plot_error()


if __name__ == '__main__':
    from time import time

    t0 = time()
    main()
    # xor_problem()
    t = time() - t0
    print(f'Done in {t} s')
