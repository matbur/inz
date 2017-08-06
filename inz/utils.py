""" Module contains common functions used in project. """

from typing import Iterable

import numpy as np


def split(iterable: Iterable, width: int, allow_missing=True):
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
