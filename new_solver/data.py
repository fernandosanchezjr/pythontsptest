import itertools
import logging
import typing as t
from operator import itemgetter

import math
import networkx as nx
import numpy as np
from dataclasses import dataclass, replace
from geoindex import utils as geo_utils

from new_solver import constants, util

logger = logging.getLogger(__name__)
COORD_DELIMITER = "NODE_COORD_SECTION"
RADIUS = 0.5

Coords = t.Tuple[float, float]
NPCoords = t.List[float]


@dataclass(frozen=True)
class Point:
    id_: int
    lon: float
    lat: float
    rad_lon: float
    rad_lat: float
    duplicates: t.Optional[t.Tuple['Point']] = None

    @property
    def coords(self) -> Coords:
        return self.lon, self.lat

    @property
    def map_coords(self) -> Coords:
        return self.lon % 360.0, self.lat

    def merge_duplicates(self, duplicates: t.Tuple['Point']) -> 'Point':
        return create_point(self.id_, self.lon, self.lat, duplicates)

    def array(self) -> NPCoords:
        return [self.lon, self.lat]

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.id_ == other.id_
        return self == other

    def distance_to(self, point):
        """
        Calculate distance in miles or kilometers between current and other
        passed point.
        """
        assert isinstance(point, Point), (
            'Other point should also be a Point instance.'
        )
        if self.coords == point.coords:
            return 0.0
        coefficient = 69.09
        theta = self.lon - point.lon

        distance = math.degrees(math.acos(
            math.sin(self.rad_lat) * math.sin(point.rad_lat) +
            math.cos(self.rad_lat) * math.cos(point.rad_lat) *
            math.cos(math.radians(theta))
        )) * coefficient

        return geo_utils.mi_to_km(distance)


def create_point(_id: int, lon: float, lat: float, duplicates: t.Optional[t.Tuple['Point']] = None) -> Point:
    return Point(_id, lon, lat, math.radians(lon), math.radians(lat), duplicates)


Segment = t.List[Coords]


@dataclass(frozen=True)
class Grid:
    lon: float
    lat: float
    points: t.List[Point]
    graph: t.Optional[nx.Graph]

    @classmethod
    def create(
        cls,
        coords: Coords,
        points: t.List[Point],
    ) -> 'Grid':
        lon, lat = coords
        return cls(lon=lon, lat=lat, points=points, graph=None)

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

    def bounds(
        self,
    ) -> t.List[Coords]:
        lon, lat = self.map_coords
        lon1, lat1 = lon - RADIUS, lat + RADIUS
        lon2, lat2 = lon + RADIUS, lat - RADIUS
        if lon1 < 0.0 and lon2 == 0.0:
            lon2 = -0.00000000001
        return [(lon1, lat1),
                (lon2, lat1),
                (lon2, lat2),
                (lon1, lat2)]

    def zoom(self):
        lon, lat = self.map_coords
        lon1, lat1 = lon - RADIUS, lat - RADIUS
        lon2, lat2 = lon + RADIUS, lat + RADIUS
        return (lon, lat), (lon1, lat1), (lon2, lat2)

    def map(self) -> t.Tuple[t.Tuple[t.List[Coords], float], t.List[Coords], t.List[Segment]]:
        return (
            (self.bounds(), RADIUS),
            [n.map_coords for n in self.graph.nodes()],
            [[a.map_coords, b.map_coords] for a, b in list(self.graph.edges())],
        )

    def set_graph(self, graph: nx.Graph) -> 'Grid':
        return replace(self, graph=graph)


def _initial_grid_coords(lon: float, lat: float) -> Coords:
    return (math.trunc(lon) + (-RADIUS if lon < 0.0 else RADIUS),
            math.trunc(lat) + (-RADIUS if lat < 0.0 else RADIUS))


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
            point = create_point(int(id_), coord_parser(lon), coord_parser(lat), None)
            points = coordinates.setdefault(point.coords, [])
            points.append(point)
        except ValueError:
            pass
    return [first.merge_duplicates(tuple(rest))
            for first, *rest in coordinates.values()]
