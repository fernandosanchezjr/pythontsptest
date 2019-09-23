import typing as t

from shapely.geometry import (MultiPolygon, Point as ShapelyPoint, Polygon)

from new_solver import data


def inner_hull(
    center: t.Union[data.Grid, data.Point],
    neighbor_points: t.Dict[data.Coords, t.Union[data.Grid, data.Point]],
    triangulation: t.List[Polygon],
) -> t.List[t.Union[data.Grid, data.Point]]:
    center_point = ShapelyPoint(*center.coords)
    polygons = MultiPolygon([tr for tr in triangulation
                             if tr.touches(center_point)])
    unique_coords = set([c for c in polygons.convex_hull.exterior.coords
                         if c != center.coords])
    return [neighbor_points[c] for c in unique_coords]
