import typing as t
from operator import itemgetter

from shapely.geometry import LineString

from new_solver import data


def inner_hull(
    center: t.Union[data.Grid, data.Point],
    neighbors: t.List[t.Union[data.Grid, data.Point]],
) -> t.List[t.Union[data.Grid, data.Point]]:
    distances = sorted(((n, n.distance_to(center)) for n in neighbors),
                       key=itemgetter(1))
    hull_grids = distances[:2]
    distances = distances[2:]
    while len(distances):
        line = LineString([g.coords for g, _ in hull_grids])
        distances = list(filter(
            (lambda og: LineString([center.coords,
                                    og[0].coords]).intersects(line) is False),
            distances))
        hull_grids.extend(distances[:1])
    return [g for g, _ in hull_grids]
