import matplotlib.pyplot as plt
import numpy as np

from inz import activation, split
from inz.data_generator import generate_data

np.random.seed(42)


class Perceptron:
    def __init__(self):
        np.random.seed(42)
        self.w = np.random.rand(3, 1) - .5
        # self.w = np.array([[0], [1.], [-.8]])

    def fit(self, x, d, n_epoch=1001, batch_size=64):
        xlen = len(x)
        assert xlen == len(d), 'Features and labels must have the same length'

        x = np.c_[[-1] * xlen, x]
        print(x[:4, :])
        xs = x.shape

        for epoch in range(n_epoch):
            if epoch % 100 == 0:
                self.plot(x, d)

            p = np.random.permutation(xlen)
            # p = np.arange(xlen)
            errors = 0

            for batch_x, batch_y in zip(split(x[p], batch_size), split(d[p], batch_size)):
                w0 = self.w
                bxs = batch_x.shape
                bys = batch_y.shape
                print(w0, 'epoch', epoch)

                y = self.predict(batch_x)
                err = batch_y - y
                errors += np.sum(np.abs(err))
                update = batch_x.T @ err
                w = w0 + update
                self.w = w

                # r = np.sum(w0 - w)

            print('errors:', errors)

            if errors == 0:
                self.plot(x, d)
                break

    def predict(self, x):
        xw = x @ self.w
        return activation.binary_step(xw)

    def plot(self, x, y):
        plt.grid()
        x1min = x[:, 1].min() - 1
        x1max = x[:, 1].max() + 1
        x2min = x[:, 2].min() - 1
        x2max = x[:, 2].max() + 1
        plt.axis([x1min, x1max, x2min, x2max])

        # plot data
        c1 = x[y.astype(bool).T[0]]
        c2 = x[~y.astype(bool).T[0]]

        plt.scatter(c1[:, 1], c1[:, 2], marker='^')
        plt.scatter(c2[:, 1], c2[:, 2], marker='v')

        # plot weights
        bias, w1, w2 = self.w.T[0]
        plt.quiver(0, 0, w1, w2, angles='xy', scale_units='xy', scale=1)

        def get_x2(x1):
            return (-w1 * x1 + bias) / w2

        plt.plot([-100, 100], [get_x2(-100), get_x2(100)], c='k')
        plt.show()

    def __str__(self):
        return f'Perceptron: {self.w}'


def main():
    c1 = generate_data(100, 1, 0, 2 + 5)
    c2 = generate_data(100, 0, -2, -2 + 5)

    data = np.array([*c1, *c2])
    np.random.shuffle(data)

    x = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    x = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    y = np.array([0, 0, 1, 0]).reshape(-1, 1)

    print(x.shape)
    print(y.shape)

    p = Perceptron()
    p.fit(x, y)


if __name__ == '__main__':
    main()
