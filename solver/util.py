import logging
import time
from os import path


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(logging.StreamHandler())


setup_logging()
logger = logging.getLogger(__name__)


def get_relative_path(module: str, path_name: str) -> str:
    return path.abspath(path.join(path.dirname(module), path_name))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.debug("%s elapsed time: %f sec",
                      method.__qualname__, (te - ts))
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
