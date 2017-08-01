import json
from pathlib import Path

import numpy as np


def gen_data(size=10, range_=100):
    x = np.random.random(size) * range_ - range_ / 2
    y = 2 * x + 1
    data = np.array(list(zip(x, y)))

    return data


def save_data():
    data = gen_data(1000).tolist()

    file = Path('./data.json')
    file.write_text(json.dumps(data))


if __name__ == '__main__':
    save_data()
