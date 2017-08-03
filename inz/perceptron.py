import matplotlib.pyplot as plt
import numpy as np


def f(s):
    return int(s >= 0)


class Perceptron:
    def __init__(self):
        np.random.seed(42)
        # self.w = np.array([[1.], [-.8]])
        self.w = np.random.rand(2, 1)

    def train(self, x, d):
        w0 = self.w
        y = f(x.dot(w0))
        err = d - y
        w = w0 + err * x.T
        self.w = w

        r = np.sum(w0 - w)
        return not r

    def __str__(self):
        return f'Perceptron :{self.w}'


def main():
    x = np.array([
        [[1, 2]],
        [[-1, 2]],
        [[0, -1]],
    ])

    y = np.array([1, 0, 0])

    print(x)
    print(y)
    print()

    xx = [i[0, 0] for i in x]
    yy = [i[0, 1] for i in x]

    p = Perceptron()
    n = 10 ** 3

    l = [0, 0, 0]

    for i in range(50):
        i %= 3
        print(p.w)
        [x1], [x2] = p.w

        plt.quiver(0, 0, x1, x2)
        plt.scatter(x1, x2)
        plt.scatter(xx, yy, c=y)
        plt.plot([-x2 * n, x2 * n], [x1 * n, -x1 * n])
        plt.grid()
        plt.axis([-3, 3, -3, 3])
        plt.text(2, 1, f'{p.w}')
        plt.show()

        l[i] = p.train(x[i], y[i])

        if all(l):
            break


if __name__ == '__main__':
    main()
