import asyncio
import logging
import typing as t
from concurrent import futures

import psutil

from solver import data, util

logger = logging.getLogger(__name__)


class Processor:
    _executor: futures.ThreadPoolExecutor
    _loop: asyncio.events.AbstractEventLoop
    data_set: data.DataSet

    def __init__(self, data_set: data.DataSet):
        self._executor = futures.ThreadPoolExecutor(
            max_workers=psutil.cpu_count(),
            thread_name_prefix="p_"
        )
        self._loop = asyncio.get_event_loop()
        self.data_set = data_set

    @classmethod
    def create(cls, file_path: str) -> 'Processor':
        return cls(data.DataSet(file_path))

    def _wait(self, awaitable):
        self._loop.run_until_complete(awaitable)

    async def execute(self, func: (), *args):
        return await asyncio.wait(
            [self._loop.run_in_executor(self._executor, func, *args)])

    async def execute_many(self, func: (), args_list: t.Iterable[t.Any]):
        tasks = [self._loop.run_in_executor(self._executor, func, *args)
                 for args in args_list]
        return await asyncio.wait(tasks)

    @staticmethod
    def _find_grid_neighbors(grid: data.Grid):
        if len(grid.points) > 1:
            return
        for point in grid.points:
            grid.get_nearest_points(point)

    @util.timeit
    def find_grid_neighbors(self):
        self._wait(self.execute_many(self._find_grid_neighbors,
                                     [[g] for g in self.data_set.grids]))


if __name__ == "__main__":
    target_path = util.get_relative_path(__file__, "../data/world.tsp")
    logger.info("Loading %s", target_path)
    processor = Processor.create(target_path)
    logger.info("Loaded %s points", len(processor.data_set.points))
    logger.info("Generated %s grids", len(processor.data_set.grids))
    processor.find_grid_neighbors()
    input("")
