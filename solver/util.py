import logging
from os import path
import time


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
        logging.debug("%s elapsed time: %f ms",
                      method.__qualname__, (te - ts) * 1000)
        return result
    return timed
