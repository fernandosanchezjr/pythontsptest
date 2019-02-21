import asyncio
import logging
import typing as t
from concurrent import futures
from operator import attrgetter

import numpy as np
import psutil

from solver import args, constants, data, graph, util

logger = logging.getLogger(__name__)


async def _processor(executor, queue, func: (), args_list: t.List[t.Any]):
    for f in executor.map(func, args_list):
        await queue.put(f)


async def _consumer(queue, chunk_count: int):
    completed_chunks = 0
    results = [None] * chunk_count
    while completed_chunks < chunk_count:
        chunk = await queue.get()
        results[completed_chunks] = chunk
        completed_chunks += 1
    return results


class BaseProcessor:
    _executor_count: int
    _executor: futures.ProcessPoolExecutor
    _loop: asyncio.events.AbstractEventLoop
    _queue: asyncio.Queue
    data_set: data.DataSet
    index: t.Optional[data.Index]

    def __init__(self, data_set: data.DataSet):
        self._executor_count = psutil.cpu_count() - 1 or 1
        p = psutil.Process()
        p.cpu_affinity(range(psutil.cpu_count()))
        self._executor = futures.ProcessPoolExecutor(
            max_workers=self._executor_count
        )
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self.data_set = data_set
        self.index = None

    def process(self, func: (), args_list: t.List[t.Any]):
        _, results = self._loop.run_until_complete(asyncio.gather(
            _processor(self._executor, self._queue, func, args_list),
            _consumer(self._queue, len(args_list))
        ))
        return results


def _subdivide(grid: data.Grid) -> data.Grid:
    grid.subdivide()
    return grid


def _start_grid_seeds(grid: data.Grid) -> data.Grid:
    terminals = list(filter(attrgetter('seed'), grid.terminals()))
    seeds: t.List[data.Point] = list(map(attrgetter('seed'),
                                         terminals))
    index = data.Index(list(grid.endpoints()))
    for seed in seeds:
        nearest = index.get_nearest(seed,
                                    min_count=constants.SEED_DISTANCES)
        nearest_segments = []
        for target, distance in nearest:
            nearest_segments.append(seed.segment_to(target, distance))
            if (not isinstance(target, data.Segment.Pointer) and
                len(nearest_segments) > constants.MIN_RESULT_COUNT):
                break
        if not nearest_segments:
            continue
        _, route = grid.sieve(seed)
        new_route = data.remove_nested_entry(route, seed)
        new_route[-1].append(*nearest_segments)
    return grid


def _find_clusters(grid: data.Grid):
    endpoints = {g.id_: g
                 for g in grid.endpoints()}
    clusters = []
    while len(endpoints):
        c = data.Cluster()
        for e in list(endpoints.values()):
            if c.empty():
                c.append(e)
                del endpoints[e.id_]
            elif c.intersects(e):
                c.append(e)
                del endpoints[e.id_]
        clusters.append(c)
    grid.set(clusters)
    return grid


class Processor(BaseProcessor):

    @classmethod
    def create(cls, file_path: str) -> 'Processor':
        return cls(data.DataSet(file_path))

    @util.timeit
    def subdivide(self):
        new_grids = self.process(_subdivide, self.data_set.grids)
        self.data_set.grids = new_grids

    @util.timeit
    def start_seeds(self):
        new_grids = self.process(_start_grid_seeds, self.data_set.grids)
        self.data_set.grids = new_grids

    @util.timeit
    def find_clusters(self):
        new_grids = self.process(_find_clusters, self.data_set.grids)
        self.data_set.grids = new_grids

    @util.timeit
    def draw_map(
        self,
        drawn_grids: t.Iterable[data.Grid] = None,
        center: data.Coords = (0, 0),
        bottom_left: data.Coords = (-180, -90),
        top_right: data.Coords = (180, 90),
    ) -> graph.Map:
        m = graph.Map(f"{self.data_set.name} map", center=center, bottom_left=bottom_left, top_right=top_right)
        drawn_grids = drawn_grids or self.data_set.grids
        grids, points, segments = [], ([], []), []
        for grid in drawn_grids:
            g, new_points, s = m.generate_data(grid)
            grids.extend(g)
            if new_points:
                x, y = new_points
                old_x, old_y = points
                points = (np.concatenate((old_x, x)),
                          np.concatenate((old_y, y)))
            segments.extend(s)
        m.draw_data(grids, points, segments)
        return m

    @staticmethod
    def show():
        graph.Map.show()


@util.timeit
def main(show_map: bool = False):
    startup_args = args.parse_args()
    logger.info("Loading %s", startup_args.datafile)
    proc = Processor.create(startup_args.datafile)
    proc.subdivide()
    proc.start_seeds()
    proc.find_clusters()
    if show_map:
        proc.draw_map()
        proc.show()
    return proc


if __name__ == "__main__":
    processor = main(True)
