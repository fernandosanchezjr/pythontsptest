import asyncio
import logging
import typing as t
from concurrent import futures

import psutil

from solver import constants, data, graph, util


logger = logging.getLogger(__name__)


class Processor:
    _executor: futures.ThreadPoolExecutor
    _loop: asyncio.events.AbstractEventLoop
    data_set: data.DataSet
    index: t.Optional[data.Index]

    def __init__(self, data_set: data.DataSet):
        self._executor = futures.ProcessPoolExecutor(
            max_workers=psutil.cpu_count() - 1
        )
        self._loop = asyncio.get_event_loop()
        self.data_set = data_set
        self.index = None

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
    def _find_grid_neighbors(grid: data.Grid) -> data.Grid:
        if len(grid.contents) <= 1:
            return grid
        for point in grid.contents:
            nearest = grid.get_nearest(point)
            logger.info("Nearest to %s in %s: %s", point, grid, nearest)
        logger.info("%s processed", grid)
        return grid

    @util.timeit
    def find_grid_neighbors(self):
        result_grids = self.wait(self.execute_many(
            self._find_grid_neighbors,
            ([g] for g in self.data_set.grids)))
        self.data_set.grids = result_grids

    @staticmethod
    def _subdivide(grid: data.Grid) -> data.Grid:
        grid.subdivide()
        return grid

    @util.timeit
    def subdivide(self):
        new_grids = self.wait(self.execute_many(
            self._subdivide,
            ([g] for g in self.data_set.grids)))
        self.data_set.grids = new_grids
        self.index = data.Index(self.data_set.grids)
        self.index.build_index()

    @util.timeit
    def find_subdivided_neighbors(self):
        if self.index is None:
            return
        point = self.data_set.grids[0]
        nearest = self.index.get_nearest(point)
        for n, distance in nearest:
            segment = point.segment(n, distance)
            logger.info("Nearest to %s: %s", point, segment)

    @util.timeit
    def draw_map(
        self,
        a: t.Optional[data.IndexEntry] = None,
        b: t.Optional[data.IndexEntry] = None
    ) -> graph.Map:
        m = graph.Map(f"{self.data_set.name} map")
        grids = self.data_set.grids
        if a or b:

            def _grid_filter(grid: data.Grid) -> bool:
                if a and a.quandrant_bearing(grid) != constants.Quadrant.Q_IV:
                    return False
                if b and b.quandrant_bearing(grid) != constants.Quadrant.Q_II:
                    return False
                return True

            grids = filter(_grid_filter, self.data_set.grids)
        m.add_grids(grids)
        m.save(f"{self.data_set.name}_map.png")
        return m

    @staticmethod
    def show():
        graph.Map.show()


if __name__ == "__main__":
    target_path = util.get_relative_path(__file__, "../data/ar9152.tsp")
    logger.info("Loading %s", target_path)
    processor = Processor.create(target_path)
    processor.subdivide()
    processor.find_subdivided_neighbors()
    # processor.draw_map(a=data.IndexEntry(data.IndexEntry.numbers.next(),
    #                                      0.0, -180.0),
    #                    b=data.IndexEntry(data.IndexEntry.numbers.next(),
    #                                      -90.0, 0.0))
    # processor.show()
    input("")
