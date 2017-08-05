from typing import Iterable


def split(iterable: Iterable, width):
    """ Generator yields iterable in parts.

    :param iterable: iterable to split
    :param width: length of each part
    """
    it = iter(iterable)
    while True:
        yield [next(it) for _ in range(width)]
