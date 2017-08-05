from typing import Iterable

import numpy as np


def split(iterable: Iterable, width):
    """ Generator yields iterable in parts.

    :param iterable: iterable to split
    :param width: length of each part
    """
    it = iter(iterable)
    while True:
        rv = [next(it) for _ in range(width)]
        if isinstance(iterable, np.ndarray):
            rv = np.array(rv)
        yield rv
