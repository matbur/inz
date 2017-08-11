import numpy as np

from inz import DNN, act, fully_connected, input_data


def main():
    np.random.seed(42)

    x = np.array([
        [1, 1], [1, 0], [0, 1], [0, 0],
        [1, 1], [1, 0], [0, 1], [0, 0],
        [1, 1], [1, 0], [0, 1], [0, 0],
    ], dtype=float)
    y = np.array([
        [1, 0], [0, 1], [0, 1], [1, 0],
        [1, 0], [0, 1], [0, 1], [1, 0],
        [1, 0], [0, 1], [0, 1], [1, 0],
    ], dtype=float)

    inp = input_data(shape=(None, 2))
    fc1 = fully_connected(inp, 3, activation=act.sigmoid)
    fc2 = fully_connected(fc1, 2, activation=act.sigmoid)

    model = DNN(fc2)
    model.fit(x, y, n_epoch=30000, batch_size=1)

    model.show('y')
    model.show('delta')
    model.show('gradient')

    model.plot_error()


if __name__ == '__main__':
    main()
