import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class DataGenerator:
    def __init__(self, size=10, xrange=100, sigma=1, fn='data.json'):
        self.size = size
        self.range = xrange
        self.sigma = sigma
        self.fn = Path(fn)

        self.data = self.gen_data()

    def gen_data(self):
        x = np.linspace(0, self.range, num=self.size)
        y = 2 * x + np.random.normal(0, self.sigma, self.size)
        return list(zip(x, y))

    def save_data(self):
        dumps = json.dumps(self.data)
        self.fn.write_text(dumps)

    def plot_data(self):
        data = zip(*self.data)
        plt.scatter(*data)
        plt.show()


if __name__ == '__main__':
    dg = DataGenerator(size=100, sigma=10)
    dg.save_data()
    dg.plot_data()
