import asyncio
import itertools
import logging
import typing as t
from concurrent import futures
from operator import itemgetter

import psutil

from solver import args, data, graph, util

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


def _subdivide_and_join(grid: data.Grid) -> data.Grid:
    grid.subdivide()
    main_graph = grid.join_graphs()
    grid.graph = main_graph
    grid.set([])

    nodes = set(main_graph.nodes())
    edges = set(main_graph.edges())
    edge_nodes = set(itertools.chain.from_iterable(edges))
    non_edge_nodes = nodes - edge_nodes

    grid.set(list(nodes))
    for node in non_edge_nodes:
        nearest = grid.get_nearest(node)
        for other, distance in nearest[:10]:
            main_graph.add_edge(node, other, weight=distance)

    for a, b in edges:
        nearest_a = [(p, d, a) for p, d in grid.get_nearest(a)]
        nearest_b = [(p, d, b) for p, d in grid.get_nearest(b)]
        nearest = sorted(nearest_a + nearest_b,
                         key=itemgetter(1))
        for other, distance, node in nearest[:10]:
            main_graph.add_edge(node, other, weight=distance)

    return grid


class Processor(BaseProcessor):

    @classmethod
    def create(cls, file_path: str) -> 'Processor':
        return cls(data.DataSet(file_path))

    @util.timeit
    def subdivide_and_join(self):
        new_grids = self.process(_subdivide_and_join, self.data_set.grids)
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
        grids, points, segments = [], [], []
        for grid in drawn_grids:
            bounds, new_points, new_segments = grid.map()
            grids.append(bounds)
            points.extend(new_points)
            segments.extend(new_segments)
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
    proc.subdivide_and_join()
    if show_map:
        proc.draw_map()
        proc.show()
    return proc


if __name__ == "__main__":
    processor = main(True)
