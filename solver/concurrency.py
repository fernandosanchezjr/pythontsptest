import asyncio
from concurrent import futures
import logging
import threading
import typing as t

import psutil

from solver import util as _


logger = logging.getLogger(__name__)


class Thread:

    loop: asyncio.BaseEventLoop
    executor: futures.ProcessPoolExecutor
    thread: t.Optional[threading.Thread] = None

    def __init__(self):
        self.executor = futures.ThreadPoolExecutor(1)
        self.loop = asyncio.new_event_loop()

    def _loop(self):
        with self.executor:
            self.loop.run_forever()

    def start(self):
        if not self.thread:
            self.thread = threading.Thread(target=self._loop,
                                           daemon=True)
            self.thread.start()

    async def execute(self, func: (), *args):
        handle = self.loop.call_soon_threadsafe(func, *args)
        return handle


def cpu_bound():
    # CPU-bound operations will block the event loop:
    # in general it is preferable to run them in a
    # process pool.
    return sum(i * i for i in range(10 ** 7))


if __name__ == "__main__":
    async def test():
        count = psutil.cpu_count()
        logger.info("count: %d", count)
        t = Thread()
        t.start()
        for _ in range(10):
            result = await t.execute(cpu_bound)
            logger.info("result: %s", result)

    asyncio.run(test())
