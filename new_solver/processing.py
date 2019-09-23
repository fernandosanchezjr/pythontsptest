import asyncio
import functools
import itertools
import logging
import typing as t
from concurrent import futures

import psutil
from shapely import geometry
from shapely import ops

from new_solver import args, constants, data, graph, inner_hull, util

logger = logging.getLogger(__name__)


async def _processor(
    executor,
    queue,
    func: (),
    args_list: t.List[t.Any],
    chunksize: int = 512,
    **kwargs: t.Dict[t.Any, t.Any]
):
    for f in executor.map(functools.partial(func, **kwargs), args_list,
                          chunksize=chunksize):
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

    def process(
        self,
        func: (),
        args_list: t.List[t.Any],
        chunksize: int = 512,
        **kwargs: t.Dict[t.Any, t.Any],
    ) -> t.List[t.Any]:
        _, results = self._loop.run_until_complete(asyncio.gather(
            _processor(self._executor, self._queue, func, args_list,
                       chunksize=chunksize, **kwargs),
            _consumer(self._queue, len(args_list))
        ))
        return results


def _subdivide_grid(grid: data.Grid) -> t.Tuple[data.Coords, t.List[data.Grid]]:
    return grid.coords, grid.subdivide()


@util.timeit
def subdivide_grids(
    proc: Processor,
    grids: t.List[data.Grid],
) -> t.Tuple[t.Dict[data.Coords, t.List[data.Grid]], t.List[data.Grid]]:
    grids_by_parent = dict(proc.process(_subdivide_grid, grids))
    all_grids = list(itertools.chain.from_iterable(grids_by_parent.values()))
    return grids_by_parent, all_grids


def _generate_grid_geometry(
    grid: data.Grid
) -> t.Tuple[geometry.LinearRing,
             geometry.LineString,
             t.List[data.Coords]]:
    return grid.map()


@util.timeit
def draw_map(
    proc: Processor,
    name: str,
    grids: t.List[data.Grid],
    center: data.Coords = (0, 0),
    bottom_left: data.Coords = (-180, -90),
    top_right: data.Coords = (180, 90),
) -> graph.Map:
    m = graph.Map(f'{name} map', center=center, bottom_left=bottom_left,
                  top_right=top_right)
    drawn_grids, hulls, points = zip(*proc.process(_generate_grid_geometry,
                                                   grids))
    grid_bounds = geometry.MultiLineString(drawn_grids)
    inner_hulls = geometry.MultiLineString(
        list(itertools.chain.from_iterable(hulls)))
    all_points = list(itertools.chain.from_iterable(points))
    m.draw_lines(grid_bounds, zorder=2.0)
    m.draw_lines(inner_hulls, zorder=3.0, colors='green')
    m.draw_points(all_points, zorder=4.0)
    return m


def _evaluate_neighbors(
    search_parameters: t.Tuple[data.Coords, t.List[data.Grid]],
    grids_by_parent: t.Optional[t.Dict[data.Coords, t.List[data.Grid]]] = None,
) -> t.List[data.Grid]:
    coords, grids = search_parameters
    parent_count = len(grids_by_parent)
    neighbor_grids = []
    center = data.Grid.create(coords, [])
    external_radius = data.GRID_RADIUS
    while (len(neighbor_grids) < constants.MIN_BIN_SEARCH_COUNT
           and parent_count > 1):
        new_neighbors = center.get_neighbors(external_radius, grids_by_parent)
        for n in new_neighbors:
            neighbor_grids.extend(n)
        external_radius += data.GRID_RADIUS

    neighbor_grids = neighbor_grids + grids
    flattened_points = [p for g in neighbor_grids for p in g.points]
    if len(grids) == 1:
        nearest = sorted([(p, p.distance_to(grids[0].points[0]))
                          for p in flattened_points if p != grids[0].points[0]],
                         key=lambda p: p[1])
        new_grid = grids[0].set_neighbors(
            [n[0] for n in nearest[:constants.MIN_BIN_SEARCH_COUNT]])
        return [new_grid]
    all_points = {
        p.coords: p for p in flattened_points
    }
    known_points = geometry.MultiPoint(list(all_points.keys()))
    polygons = ops.triangulate(known_points)
    result_grids = []
    for g in grids:
        new_g = g.set_neighbors(inner_hull.inner_hull(g.points[0], all_points,
                                                      polygons))
        result_grids.append(new_g)
    return result_grids


@util.timeit
def find_nearest_neighbors(
    proc: Processor,
    grids_by_parent: t.Dict[data.Coords, t.List[data.Grid]],
):
    results = proc.process(_evaluate_neighbors, list(grids_by_parent.items()),
                           grids_by_parent=grids_by_parent,
                           chunksize=32)
    return list(itertools.chain.from_iterable(results))


@util.timeit
def load(show_map: bool = False):
    startup_args = args.parse_args()
    logger.info('Loading %s', startup_args.datafile)
    name, grids = data.load_datafile(startup_args.datafile)
    proc = Processor()
    grids_by_parent, grids = subdivide_grids(proc, grids)
    logger.info('Subdivided into %s grids', len(grids))
    grids = find_nearest_neighbors(proc, grids_by_parent)
    if show_map:
        m = draw_map(proc, name, grids)
        m.show()
    return grids


if __name__ == '__main__':
    load(True)
