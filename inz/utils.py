""" Module contains common functions used in project. """

from itertools import islice
from typing import Iterable

import numpy as np

from .layers import Layer


def old_split(iterable: Iterable, width: int, allow_missing=True):
    """ Generator yields iterable in parts.

    :param iterable: iterable to split
    :param width: length of each part
    :param allow_missing: if True last part may be smaller
    """
    it = iter(iterable)
    flag = True
    while flag:
        retval = []
        flag = False
        for _ in range(width):
            try:
                retval.append(next(it))
            except StopIteration:
                if not allow_missing:
                    return
                break
        else:
            flag = True

        if not retval:
            return

        if isinstance(iterable, np.ndarray):
            retval = np.array(retval)

        yield retval


def split(iterable: Iterable, width: int, allow_missing=True):
    """ Generator yields iterable in parts.

    :param iterable: iterable to split
    :param width: length of each part
    :param allow_missing: if True last part may be smaller
    """
    it = iter(iterable)
    while True:
        retval = list(islice(it, width))

        if not retval:
            return
        if len(retval) != width and not allow_missing:
            return

        if isinstance(iterable, np.ndarray):
            retval = np.array(retval)

        yield retval


def it(network: Layer, attr, i=0):
    if i == 0:
        print()
        print(attr)
    values = getattr(network, attr)
    previous = network.previous

    if previous is None:
        return
    shape = values.shape if values is not None else None
    print('Layer: {}, i: {} {}.shape: {}'.format(network.id, i, attr, shape))
    print(values)
    it(previous, attr, i + 1)
