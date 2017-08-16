from time import sleep

import numpy as np

from inz import DNN, fully_connected, input_data


def main():
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

    inp = input_data(shape=(None, 2))
    fc1 = fully_connected(inp, 3)
    # fc2 = fully_connected(fc1, 5)
    # fc3 = fully_connected(fc2, 4)
    fc4 = fully_connected(fc1, 2)
    net = fc4

    model = DNN(net)
    # model.fit(x, y, n_epoch=1, batch_size=1)

    sleep(.1)

    # model.show(['W', 'b', 'y', 'delta', 'gradient'], not False)

    model.show(['W', 'b'], not False)
    model.load('model.json')
    model.show(['W', 'b'], not False)

    # model.plot_error()


if __name__ == '__main__':
    main()
