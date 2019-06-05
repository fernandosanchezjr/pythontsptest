import itertools
import logging
import time
from os import path

logger = logging.getLogger(__name__)


def get_relative_path(module: str, path_name: str) -> str:
    return path.abspath(path.join(path.dirname(module), path_name))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.debug("%s elapsed time: %f sec", method.__qualname__, (te - ts))
        return result

    return timed


class Numbers:
    current: int

    def __init__(self):
        self.current = 0

    def __iter__(self) -> 'Numbers':
        return self

    def __next__(self) -> int:
        self.current += 1
        return self.current

    def next(self) -> int:
        return next(self)


def partition(pred, iterable):
    """Use a predicate to partition entries into false entries and true entries"""
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)
