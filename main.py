from time import sleep

import numpy as np

from inz import Model, fully_connected, input_data


def main():
    np.random.seed(42)

    x = np.array([
        [1, 1], [1, 0], [0, 1], [0, 0],
    ], dtype=float)
    y = np.array([
        [0, 1], [0, 1], [0, 1], [1, 0],
    ], dtype=float)
    # y = np.array([
    #     [1], [1], [1], [0],
    # ], dtype=float)

    inp = input_data(shape=(None, 2))
    # fc1 = fully_connected(inp, 3)
    # fc2 = fully_connected(fc1, 5)
    # fc3 = fully_connected(fc2, 4)
    fc4 = fully_connected(inp, 2, activation='sigmoid')
    net = fc4

    model = Model(net)

    sleep(.1)

    model.show(['W', 'b', 'y', 'delta', 'gradient'], not False)

    print(f'x = {x}')
    model.predict(x)
    model.show(['W', 'b', 'y'], not False)

    print('*' * 10)

    model.fit(x, y, n_epoch=1000, batch_size=2)
    # model.show(['W', 'b', 'y', 'delta', 'gradient'], not False)
    model.predict(x)
    model.show(['W', 'b', 'y'], not False)


    # model.plot_error()


def foo():
    x = np.array([
        [1, 1], [1, 0], [0, 1], [0, 0],
    ], dtype=float)
    y = np.array([
        [0, 1], [0, 1], [0, 1], [1, 0],
    ], dtype=float)

    inp = input_data((None, 2))
    fc1 = fully_connected(inp, 3, activation='sigmoid')
    fc2 = fully_connected(fc1, 2, activation='sigmoid')

    dnn = Model(fc2)
    # dnn.fit(x, y)

    print(dnn.predict(x))
    print(y)


if __name__ == '__main__':
    # main()
    foo()
