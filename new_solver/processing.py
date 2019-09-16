import asyncio
import functools
import itertools
import logging
import typing as t
from concurrent import futures
from operator import itemgetter

import networkx as nx
import numpy as np
import psutil
from shapely import geometry

from new_solver import (args, constants, convex_hull, data, graph, inner_hull,
                        util)

logger = logging.getLogger(__name__)


async def _processor(
    executor,
    queue,
    func: (),
    args_list: t.List[t.Any],
    **kwargs: t.Dict[t.Any, t.Any]
):
    for f in executor.map(functools.partial(func, **kwargs), args_list):
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
        **kwargs: t.Dict[t.Any, t.Any],
    ) -> t.List[t.Any]:
        _, results = self._loop.run_until_complete(asyncio.gather(
            _processor(self._executor, self._queue, func, args_list, **kwargs),
            _consumer(self._queue, len(args_list))
        ))
        return results


def extract_hull(
    points: t.List[data.Point]
) -> t.Tuple[t.List[data.Point], t.List[data.Point]]:
    all_points = [(p.array(), p) for p in points]
    hull_coords = list(np.array(convex_hull.convex_hull(
        np.array(list(map(itemgetter(0), all_points))))).tolist())

    hull_matches = sorted(((coords in hull_coords, coords, p)
                           for coords, p in all_points), key=itemgetter(0))
    grouped_points = {present: list(matches)
                      for present, matches in
                      itertools.groupby(hull_matches, key=itemgetter(0))}

    hull_points = sorted(grouped_points.get(True, []),
                         key=lambda p: hull_coords.index(p[1]))

    if len(hull_points) > 2:
        hull_points.append(hull_points[0])

    non_hull_points = grouped_points.get(False, [])

    return (list(map(itemgetter(2), hull_points)),
            list(map(itemgetter(2), non_hull_points)))


def calculate_hulls(grid: data.Grid) -> data.Grid:
    g = nx.Graph()
    all_points = grid.points
    depth = 1
    all_hulls = []
    while len(all_points) > 2:
        hull, all_points = extract_hull(all_points)
        if not hull:
            break
        if len(all_hulls) > 0:
            previous_hull = all_hulls[-1:][0]
            current_hull_polygon = geometry.Polygon([c.coords for c in hull])
            for p in previous_hull:
                for c in hull:
                    line = geometry.LineString([p.coords, c.coords])
                    if not line.crosses(current_hull_polygon):
                        g.add_edge(p, c, weight=p.distance_to(c),
                                   depth=depth, single=False)
        all_hulls.append(hull)
        for a, b in zip(hull, hull[1:]):
            g.add_edge(a, b, weight=a.distance_to(b), depth=depth,
                       single=False)
        depth += 1

    if len(all_points) == 2:
        a, b = all_points
        g.add_edge(a, b, weight=a.distance_to(b), depth=depth, single=False)
        if len(all_hulls) > 0:
            previous_hull = all_hulls[-1:][0]
            current_hull_polygon = geometry.LineString([a.coords, b.coords])
            for h in previous_hull:
                for p in all_points:
                    line = geometry.LineString([h.coords, p.coords])
                    if not line.crosses(current_hull_polygon):
                        g.add_edge(h, p, weight=h.distance_to(p),
                                   depth=depth, single=False)
    elif len(all_points) == 1:
        p = all_points[0]
        g.add_node(p, depth=depth, single=True)
        if len(all_hulls) > 0:
            previous_hull = all_hulls[-1:][0]
            for h in previous_hull:
                g.add_edge(h, p, weight=h.distance_to(p),
                           depth=depth, single=False)
    elif len(all_points) == 0 and len(all_hulls):
        last_hull = all_hulls[-1:][0]
        if len(last_hull) > 0:
            previous_hull = all_hulls[-1:][0]
            for p in previous_hull:
                for c in previous_hull:
                    if not g.has_edge(p, c):
                        g.add_edge(p, c, weight=p.distance_to(c),
                                   depth=depth, single=False)
    all_hulls.append(all_points)

    return grid.set_graph(g).set_hull(all_hulls[0]).set_depth(depth)


def build_search_graph(
    params: t.Tuple[data.Grid, t.Dict[data.Coords, t.List[data.Point]]]
) -> data.Grid:
    grid, all_grid_coords = params
    radius = data.GRID_RADIUS
    all_hulls = []
    neighbor_hull_points = []
    while len(neighbor_hull_points) < 8:
        bounding_grids = grid.bounding_grids(radius=radius)
        for b in bounding_grids:
            if b in all_grid_coords:
                neighbor_hull = all_grid_coords[b]
                neighbor_hull_size = len(neighbor_hull)
                if neighbor_hull_size > 2:
                    all_hulls.append(geometry.Polygon(
                        [p.coords for p in neighbor_hull]))
                elif neighbor_hull_size == 2:
                    all_hulls.append(geometry.LineString(
                        [p.coords for p in neighbor_hull]))
                neighbor_hull_points.extend(neighbor_hull)
        radius += data.GRID_RADIUS
    local_hull_size = len(grid.hull)
    if local_hull_size > 2:
        all_hulls.append(geometry.Polygon(
            [p.coords for p in grid.hull]))
    elif local_hull_size == 2:
        all_hulls.append(geometry.LineString(
            [p.coords for p in grid.hull]))
    neighbor_points = []
    for h in grid.hull:
        for n in neighbor_hull_points:
            line = geometry.LineString([h.coords, n.coords])
            if not any([line.crosses(ah) for ah in all_hulls]):
                grid.graph.add_edge(h, n, weight=h.distance_to(n), depth=0,
                                    external=True)
                neighbor_points.append(n)
    return grid.set_neighbors(neighbor_points)


@util.timeit
def process_grids(
    proc: Processor, grids: t.List[data.Grid]
) -> t.List[data.Grid]:
    grids = proc.process(calculate_hulls, grids)
    all_grid_coords = {g.coords: g.hull for g in grids}
    grids = proc.process(build_search_graph, [(g, all_grid_coords)
                                              for g in grids])

    return grids


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


def generate_grid_geometry(
    grid: data.Grid
) -> t.Tuple[t.Tuple[t.List[data.Coords], float], t.List[data.Coords],
             t.List[data.Segment], t.List[data.Segment]]:
    return grid.map()


@util.timeit
def draw_map(
    proc: Processor,
    name: str,
    grids: t.List[data.Grid],
    show_external: bool = True,
    center: data.Coords = (0, 0),
    bottom_left: data.Coords = (-180, -90),
    top_right: data.Coords = (180, 90),
) -> graph.Map:
    m = graph.Map(f'{name} map', center=center, bottom_left=bottom_left,
                  top_right=top_right)
    drawn_grids, points, internal, external = zip(*proc.process(
        generate_grid_geometry, grids))
    m.draw_grids(drawn_grids,
                 list(itertools.chain.from_iterable(points)),
                 list(itertools.chain.from_iterable(internal)),
                 list(itertools.chain.from_iterable(external))
                 if show_external else None)
    return m


def generate_map_grid(
    grid: data.Grid
) -> t.Tuple[t.Tuple[t.List[data.Coords], float], t.List[data.Coords]]:
    return grid.map_grid()


@util.timeit
def draw_grid_map(
    proc: Processor,
    name: str,
    grids: t.List[data.Grid],
    center: data.Coords = (0, 0),
    bottom_left: data.Coords = (-180, -90),
    top_right: data.Coords = (180, 90),
) -> graph.Map:
    m = graph.Map(f'{name} map', center=center, bottom_left=bottom_left,
                  top_right=top_right)
    drawn_grids, points = zip(*proc.process(generate_map_grid, grids))
    m.draw_grids(drawn_grids, list(itertools.chain.from_iterable(points)),
                 [], [])
    return m


def _evaluate_neighbors(
    search_parameters: t.Tuple[data.Coords, t.List[data.Grid]],
    grids_by_parent: t.Optional[t.Dict[data.Coords, t.List[data.Grid]]] = None,
) -> t.Tuple[data.Coords, t.List[data.Grid]]:
    coords, grids = search_parameters
    parent_count = len(grids_by_parent)
    for g in grids:
        external_radius = data.GRID_RADIUS
        if len(grids) == 1:
            logger.info('Lone grid %s', g.coords)
        candidate_neighbors = list(filter(lambda ng: ng != g, grids))
        while ((len(candidate_neighbors) < constants.MIN_RESULT_COUNT
                and parent_count > 1) or external_radius == data.GRID_RADIUS):
            neighbors = g.get_neighbors(external_radius, grids_by_parent)
            candidate_neighbors.extend(neighbors)
            external_radius += data.GRID_RADIUS
        neighbors = inner_hull.inner_hull(g, neighbors)
    return search_parameters


@util.timeit
def find_nearest_neighbors(
    proc: Processor,
    grids_by_parent: t.Dict[data.Coords, t.List[data.Grid]],
):
    result = proc.process(_evaluate_neighbors, list(grids_by_parent.items()),
                          grids_by_parent=grids_by_parent)


@util.timeit
def load(show_map: bool = False):
    startup_args = args.parse_args()
    logger.info('Loading %s', startup_args.datafile)
    name, grids = data.load_datafile(startup_args.datafile)
    proc = Processor()
    grids_by_parent, grids = subdivide_grids(proc, grids)
    logger.info('Subdivided into %s grids', len(grids))
    find_nearest_neighbors(proc, grids_by_parent)
    if show_map:
        m = draw_grid_map(proc, name, grids)
        m.show()
    return grids


if __name__ == '__main__':
    load(True)
