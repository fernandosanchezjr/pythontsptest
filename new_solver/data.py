import itertools
import logging
import math
import typing as t
from operator import itemgetter

import networkx as nx
import numpy as np
from dataclasses import dataclass, replace
from geoindex import utils as geo_utils

from new_solver import constants, util

logger = logging.getLogger(__name__)
COORD_DELIMITER = "NODE_COORD_SECTION"
RADIUS = 0.5
GRID_RADIUS = 1.0

Coords = t.Tuple[float, float]
NPCoords = t.List[float]


def fix_longitude(lon: float) -> float:
    if lon > 180.0:
        return lon % -180.0
    elif lon < 180.0:
        return lon % 180.0
    return lon


def fix_latitude(lat: float) -> float:
    if lat > 90.0:
        return lat % -90.0
    if lat < -90.0:
        return lat % 90.0
    return lat


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


def create_point(
    _id: int,
    lon: float,
    lat: float,
    duplicates: t.Optional[t.Tuple['Point']] = None
) -> Point:
    return Point(_id, lon, lat, math.radians(lon), math.radians(lat),
                 duplicates)


Segment = t.List[Coords]

DataEdge = t.Tuple[Point, Point, t.Dict[str, t.Any]]


@dataclass(frozen=True)
class Grid:
    lon: float
    lat: float
    points: t.List[Point]
    graph: t.Optional[nx.Graph]
    hull: t.Optional[t.List[Point]]
    neighbors: t.Optional[t.List[Point]]
    parent_coords: Coords
    depth: int = 0
    radius: float = constants.INITIAL_RADIUS

    @classmethod
    def create(
        cls,
        coords: Coords,
        points: t.List[Point],
    ) -> 'Grid':
        lon, lat = coords
        return cls(lon=lon, lat=lat, points=points, graph=None, hull=None,
                   neighbors=None, parent_coords=(lon, lat))

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
        radius: t.Optional[float] = None,
        to_map: bool = True,
    ) -> t.List[Coords]:
        _radius = radius or self.radius
        lon, lat = self.map_coords if to_map else self.coords
        lon1, lat1 = fix_longitude(lon - _radius), fix_latitude(lat + _radius)
        lon2, lat2 = fix_longitude(lon + _radius), fix_latitude(lat - _radius)
        if lon1 < 0.0 and lon2 == 0.0:
            lon2 = -0.00000000001
        elif lon1 == 180.0:
            lon1 = -179.99999999999
        elif lon2 == -180.0:
            lon2 = -179.99999999999
        return [(lon1, lat1),
                (lon2, lat1),
                (lon2, lat2),
                (lon1, lat2)]

    def sub_quadrants(self) -> (t.Tuple[float, t.Tuple[Coords,
                                                       Coords,
                                                       Coords,
                                                       Coords]]):
        new_radius = self.radius / 2.0
        return (new_radius,
                ((self.lon + new_radius, self.lat + new_radius),
                 (self.lon - new_radius, self.lat + new_radius),
                 (self.lon - new_radius, self.lat - new_radius),
                 (self.lon + new_radius, self.lat - new_radius)))

    def bounding_grids(self, radius: t.Optional[float] = None) -> t.Set[Coords]:
        (lon1, lat1), _, (lon2, lat2), _ = self.bounds(radius=radius,
                                                       to_map=False)
        lon_range = np.arange(lon1, lon2 + GRID_RADIUS, GRID_RADIUS).tolist()
        lat_range = np.arange(lat2, lat1 + GRID_RADIUS, GRID_RADIUS).tolist()
        left = list(zip([lon_range[0]] * len(lat_range), lat_range))
        right = list(zip(lon_range[-1:] * len(lat_range), lat_range))
        top = list(zip(lon_range, [lat_range[0]] * len(lon_range)))
        bottom = list(zip(lon_range, lat_range[-1:] * len(lon_range)))
        return set(left + top + right + bottom)

    def redistribute(
        self,
        point: Point,
        quadrants: t.Tuple[Coords, Coords, Coords, Coords],
    ):
        bearing = self.quandrant_bearing(point.lon, point.lat)
        return (quadrants[bearing], bearing), point

    def subdivide(
        self,
        parent_coords: t.Optional[Coords] = None
    ) -> t.List['Grid']:
        if not parent_coords:
            parent_coords = self.coords
        if len(self.points) <= constants.MAX_GRID_DENSITY:
            return [self]
        new_radius, quadrant_coords = self.sub_quadrants()
        new_depth = self.depth + 1
        redistributed_points = (self.redistribute(p, quadrant_coords)
                                for p in self.points)
        grouped_points = itertools.groupby(sorted(redistributed_points,
                                                  key=itemgetter(0)),
                                           key=itemgetter(0))
        new_contents = [Grid(lon, lat, list(map(itemgetter(1), points)),
                             radius=new_radius, depth=new_depth, graph=None,
                             hull=None, neighbors=None,
                             parent_coords=parent_coords)
                        for ((lon, lat), bearing), points in grouped_points]
        results = []
        for c in new_contents:
            if isinstance(c, Grid):
                results.extend(c.subdivide(parent_coords=parent_coords))

        return results

    def zoom(self, radius: t.Optional[float] = None):
        _radius = radius or self.radius
        lon, lat = self.map_coords
        lon1, lat1 = lon - _radius, lat - _radius
        lon2, lat2 = lon + _radius, lat + _radius
        return (lon, lat), (lon1, lat1), (lon2, lat2)

    def map(self) -> t.Tuple[t.Tuple[t.List[Coords], float], t.List[Coords],
                             t.List[Segment], t.List[Segment]]:
        internal, external = util.partition(lambda edge: edge[2].get('external'),
                                            self.graph.edges(data=True))
        return (
            (self.bounds(), RADIUS),
            [n.map_coords for n in self.graph.nodes()],
            [[a.map_coords, b.map_coords] for a, b, _ in list(internal)],
            [[a.map_coords, b.map_coords] for a, b, _ in list(external)],
        )

    def map_grid(self) -> t.Tuple[t.Tuple[t.List[Coords], float],
                                  t.List[Coords]]:
        return (
            (self.bounds(), RADIUS),
            [p.map_coords for p in self.points],
        )

    def set_graph(self, graph: nx.Graph) -> 'Grid':
        return replace(self, graph=graph)

    def set_hull(self, hull: t.Optional[t.List[Point]]) -> 'Grid':
        return replace(self, hull=hull)

    def set_neighbors(self, neighbors: t.Optional[t.List[Point]]) -> 'Grid':
        return replace(self, neighbors=neighbors)

    def set_depth(self, depth: int) -> 'Grid':
        return replace(self, depth=depth)


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
        edge_weight_type: constants.EdgeWeightType = meta_data.get(
            "edge_weight_type", constants.EdgeWeightType.EUC_2D)
        points = _read_points(fh, edge_weight_type)

    points_by_grid = sorted(((_initial_grid_coords(p.lon, p.lat), p)
                             for p in points),
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
            point = create_point(int(id_), coord_parser(lon),
                                 coord_parser(lat), None)
            points = coordinates.setdefault(point.coords, [])
            points.append(point)
        except ValueError:
            pass
    return [first.merge_duplicates(tuple(rest))
            for first, *rest in coordinates.values()]
