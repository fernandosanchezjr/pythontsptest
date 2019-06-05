import asyncio
import itertools
import logging
import typing as t
from concurrent import futures
from operator import itemgetter

import networkx as nx
import numpy as np
import psutil

from new_solver import args, convex_hull, graph, data, util

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


class Processor:
    _executor_count: int
    _executor: futures.ProcessPoolExecutor
    _loop: asyncio.events.AbstractEventLoop
    _queue: asyncio.Queue

    def __init__(self):
        self._executor_count = psutil.cpu_count()
        p = psutil.Process()
        p.cpu_affinity(range(psutil.cpu_count()))
        self._executor = futures.ProcessPoolExecutor(
            max_workers=self._executor_count
        )
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self.index = None

    def process(self, func: (), args_list: t.List[t.Any]) -> t.List[t.Any]:
        _, results = self._loop.run_until_complete(asyncio.gather(
            _processor(self._executor, self._queue, func, args_list),
            _consumer(self._queue, len(args_list))
        ))
        return results


def extract_hull(points: t.List[data.Point]) -> t.Tuple[t.List[data.Point], t.List[data.Point]]:
    all_points = [(p.array(), p) for p in points]
    hull_coords = list(np.array(convex_hull.convex_hull(np.array(list(map(itemgetter(0), all_points))))).tolist())

    hull_matches = sorted(((coords in hull_coords, coords, p) for coords, p in all_points), key=itemgetter(0))
    grouped_points = {present: list(matches) for present, matches in itertools.groupby(hull_matches, key=itemgetter(0))}

    hull_points = sorted(grouped_points.get(True, []), key=lambda p: hull_coords.index(p[1]))

    if len(hull_points) > 2:
        hull_points.append(hull_points[0])

    non_hull_points = grouped_points.get(False, [])

    return list(map(itemgetter(2), hull_points)), list(map(itemgetter(2), non_hull_points))


def _do_convex_hull(grid: data.Grid) -> data.Grid:
    g = nx.Graph()
    all_points = grid.points
    depth = 0
    while len(all_points) > 2:
        hull, all_points = extract_hull(all_points)
        if not hull:
            break
        for a, b in zip(hull, hull[1:]):
            g.add_edge(a, b, weight=a.distance_to(b), depth=depth, single=False)
        depth += 1

    if len(all_points) == 2:
        a, b = all_points
        g.add_edge(a, b, weight=a.distance_to(b), depth=depth, single=False)
        depth += 1

    elif len(all_points) == 1:
        g.add_node(all_points[0], depth=depth, single=True)

    return grid.set_graph(g)


def convex_hulls(processor: Processor, grids: t.List[data.Grid]) -> t.List[data.Grid]:
    return processor.process(_do_convex_hull, grids)


@util.timeit
def process_grids(grids: t.List[data.Grid]) -> t.List[data.Grid]:
    proc = Processor()
    return convex_hulls(proc, grids)


@util.timeit
def draw_map(
    name: str,
    grids: t.Iterable[data.Grid],
    center: data.Coords = (0, 0),
    bottom_left: data.Coords = (-180, -90),
    top_right: data.Coords = (180, 90),
) -> graph.Map:
    m = graph.Map(f"{name} map", center=center, bottom_left=bottom_left, top_right=top_right)
    drawn_grids, points, segments = [], [], []
    for grid in grids:
        bounds, new_points, new_segments = grid.map()
        drawn_grids.append(bounds)
        points.extend(new_points)
        segments.extend(new_segments)
    m.draw_data(drawn_grids, points, segments)
    return m


@util.timeit
def load(show_map: bool = False):
    startup_args = args.parse_args()
    logger.info("Loading %s", startup_args.datafile)
    name, grids = data.load_datafile(startup_args.datafile)
    grids = process_grids(grids)
    if show_map:
        m = draw_map(name, grids)
        m.show()
    return grids


if __name__ == "__main__":
    load(True)
