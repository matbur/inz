import matplotlib.pyplot as plt
import numpy as np

from inz import split
from inz.data_generator import generate_data

plt

np.random.seed(42)


def f(s):
    return s >= 0


c1 = generate_data(100, 1, 2, 2)
c2 = generate_data(100, 0, -2, -2)

data = np.array([*c1, *c2])
np.random.shuffle(data)

x = data[:, :-1]
y = data[:, -1].reshape(-1, 1)


class Perceptron:
    def __init__(self):
        np.random.seed(42)
        # self.w = np.array([[1.], [-.8]])
        self.w = np.random.rand(2, 1) - .5

    def fit(self, x, d, n_epoch=10, batch_size=64):
        xlen = len(x)
        assert xlen == len(d), 'Features and labels must have the same length'

        for epoch in range(n_epoch):
            p = np.random.permutation(xlen)
            self.plot()
            for batch_x, batch_y in zip(split(x[p], batch_size), split(d[p], batch_size)):
                w0 = self.w
                bxs = batch_x.shape
                bys = batch_y.shape
                print(w0, 'epoch', epoch)

                y = self.predict(batch_x)
                err = batch_y - y
                w = w0 + batch_x.T @ err
                self.w = w

                # r = np.sum(w0 - w)

    def predict(self, x):
        return f(x @ self.w)

    def plot(self):
        plt.grid()
        plt.axis([-10, 10, -10, 10])

        # plot data
        plt.scatter(*c1[:, :-1].T)
        plt.scatter(*c2[:, :-1].T)

        x1, x2 = self.w.T[0]

        plt.scatter(x1, x2, marker='s')
        plt.plot([0, x1], [0, x2], c='k')
        n = 10 ** 2
        plt.plot(
            [-x2, x2],
            [x1, -x1],
            c='pink'
        )
        plt.show()

    def __str__(self):
        return f'Perceptron :{self.w}'


def main():
    # x = np.array([
    #     [1, 2],
    #     [-1, 2],
    #     [0, -1],
    # ])
    # y = np.array([1, 0, 0]).reshape(-1, 1)

    print(x.shape)
    print(y.shape)

    p = Perceptron()
    p.fit(x, y, n_epoch=3)


if __name__ == '__main__':
    main()
