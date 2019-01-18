import asyncio
import logging
import typing as t
from concurrent import futures

import psutil

from solver import data, util, graph


logger = logging.getLogger(__name__)


class Processor:
    _executor: futures.ThreadPoolExecutor
    _loop: asyncio.events.AbstractEventLoop
    data_set: data.DataSet

    def __init__(self, data_set: data.DataSet):
        self._executor = futures.ProcessPoolExecutor(
            max_workers=psutil.cpu_count() - 1
        )
        self._loop = asyncio.get_event_loop()
        self.data_set = data_set

    @classmethod
    def create(cls, file_path: str) -> 'Processor':
        return cls(data.DataSet(file_path))

    def wait(self, awaitable):
        results, _ = self._loop.run_until_complete(awaitable)
        return [r.result() for r in results]

    async def execute(self, func: (), *args):
        return await asyncio.wait(
            [self._loop.run_in_executor(self._executor, func, *args)])

    async def execute_many(self, func: (), args_list: t.Iterable[t.Any]):
        tasks = [self._loop.run_in_executor(self._executor, func, *args)
                 for args in args_list]
        return await asyncio.wait(tasks)

    @staticmethod
    def _find_grid_neighbors(grid: data.Grid):
        if len(grid.contents) <= 1:
            return grid
        for point in grid.contents:
            nearest = grid.get_nearest_points(point)
            # logger.info("Nearest to %s in %s: %s", point, grid, nearest)
        logger.info("%s processed", grid)
        return grid

    @util.timeit
    def find_grid_neighbors(self):
        result_grids = self.wait(self.execute_many(
            self._find_grid_neighbors,
            ([g] for g in self.data_set.grids)))
        self.data_set.grids = result_grids

    @util.timeit
    def draw_map(self) -> graph.Map:
        m = graph.Map(f"{self.data_set.name} map")
        m.add_grids(self.data_set.grids)
        m.add_points(self.data_set.grids, color='yellow', markersize=2)
        for grid in self.data_set.grids:
            m.add_points(grid.contents)
        m.save(f"{self.data_set.name}_map.png")
        return m

    @staticmethod
    def show():
        graph.Map.show()


if __name__ == "__main__":
    target_path = util.get_relative_path(__file__, "../data/world.tsp")
    logger.info("Loading %s", target_path)
    processor = Processor.create(target_path)
    # processor.find_grid_neighbors()
    processor.draw_map()
    processor.show()
