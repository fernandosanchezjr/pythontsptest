import itertools
import logging
import math
import typing as t
from operator import itemgetter

import numpy as np
from dataclasses import dataclass

from new_solver import constants, util
from new_solver.convex_hull import convex_hull

logger = logging.getLogger(__name__)
COORD_DELIMITER = "NODE_COORD_SECTION"

Coords = t.Tuple[float, float]
NPCoords = t.List[float]


@dataclass(frozen=True)
class Point:
    id: int
    lon: float
    lat: float
    duplicates: t.List['Point']

    @property
    def coords(self) -> Coords:
        return self.lon, self.lat

    @property
    def map_coords(self) -> Coords:
        return self.lon % 360.0, self.lat

    def merge_duplicates(self, duplicates: t.List['Point']) -> 'Point':
        return Point(self.id, self.lon, self.lat, duplicates)

    def array(self) -> NPCoords:
        return [self.lon, self.lat]


@dataclass(frozen=True)
class Grid:
    lon: float
    lat: float
    points: t.List[Point]
    hull: t.Optional[np.ndarray]

    @classmethod
    def create(
        cls,
        coords: Coords,
        points: t.List[Point],
        hull: t.Optional[np.ndarray] = None
    ) -> 'Grid':
        lon, lat = coords
        return cls(lon=lon, lat=lat, points=points, hull=hull)

    def quandrant_bearing(self, lon: float, lat: float) -> constants.Quadrant:
        if lat >= self.lat:
            if lon >= self.lon:
                return constants.Quadrant.Q_I
            else:
                return constants.Quadrant.Q_II
        else:
            if lon >= self.lon:
                return constants.Quadrant.Q_IV
            else:
                return constants.Quadrant.Q_III

    @property
    def coords(self) -> Coords:
        return self.lon, self.lat

    @property
    def map_coords(self) -> Coords:
        return self.lon % 360.0, self.lat

    def array(self) -> np.ndarray:
        return np.array([p.array() for p in self.points])

    def calculate_hull(self) -> 'Grid':
        hull = np.array(convex_hull(self.array()))
        return Grid.create(self.coords, self.points, hull)


def _initial_grid_coords(lon: float, lat: float) -> Coords:
    return (math.trunc(lon) + (-0.5 if lon < 0.0 else 0.5),
            math.trunc(lat) + (-0.5 if lat < 0.0 else 0.5))


def _euc_2d_parser(coord: str) -> float:
    return float(coord) * -0.001


@util.timeit
def load_datafile(path_name):
    with open(path_name) as fh:
        meta_data = _read_metadata(fh)
        name = meta_data.get("name")
        edge_weight_type: constants.EdgeWeightType = meta_data.get("edge_weight_type", constants.EdgeWeightType.EUC_2D)
        points = _read_points(fh, edge_weight_type)

    points_by_grid = sorted(((_initial_grid_coords(p.lon, p.lat), p) for p in points),
                            key=itemgetter(0))
    grids = [Grid.create(g, list(map(itemgetter(1), ps)))
             for (g, ps) in itertools.groupby(points_by_grid, itemgetter(0))]
    logger.info("Loaded %s points", len(points))
    logger.info("Loaded %s grids", len(grids))
    return name, grids


def _read_metadata(fh: t.TextIO) -> t.Mapping[str, str]:
    meta_data = {}
    for read_line in fh:
        if COORD_DELIMITER in read_line:
            break
        field, value = read_line.strip().split(" : ")
        meta_data[field.lower()] = value
    return meta_data


def _read_points(
    fh: t.TextIO,
    edge_weight_type: constants.EdgeWeightType
):
    coord_parser = float
    if edge_weight_type == constants.EdgeWeightType.EUC_2D:
        coord_parser = _euc_2d_parser
    coordinates = {}
    for read_line in fh:
        try:
            id_, lat, lon = read_line.strip().split(" ")
            point = Point(int(id_), coord_parser(lon), coord_parser(lat), None)
            points = coordinates.setdefault(point.coords, [])
            points.append(point)
        except ValueError:
            pass
    return [first.merge_duplicates(rest)
            for first, *rest in coordinates.values()]
