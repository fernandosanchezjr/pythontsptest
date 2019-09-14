import enum
import itertools
import logging
import math
import typing as t
from operator import itemgetter
from os import path

import networkx as nx
from geoindex import GeoGridIndex, GeoPoint
from geoindex.geo_grid_index import GEO_HASH_GRID_SIZE

from solver import constants, util

logger = logging.getLogger(__name__)
COORD_DELIMITER = "NODE_COORD_SECTION"

NUMBERS: t.ClassVar[util.Numbers] = util.Numbers()


class EdgeWeightType(str, enum.Enum):
    EUC_2D = "EUC_2D"
    GEOM = "GEOM"


Coords = t.Tuple[float, float]


def _initial_grid_coords(lat: float, lon: float) -> Coords:
    return (math.trunc(lat) + (-0.5 if lat < 0.0 else 0.5),
            math.trunc(lon) + (-0.5 if lon < 0.0 else 0.5))


def _euc_2d_parser(coord: str) -> float:
    return float(coord) * -0.001


def xy(latitude: float, longitude: float) -> Coords:
    return longitude, latitude


class Indexable:
    id_: int


class IndexPoint(GeoPoint, Indexable):

    def __init__(self, id_: int, latitude: float, longitude: float):
        self.id_ = id_
        super().__init__(latitude, longitude)

    def __hash__(self) -> int:
        return self.id_

    def __eq__(self, other):
        if isinstance(other, IndexPoint):
            return self.id_ == other.id_
        return False

    def __repr__(self):
        return f'{self.__class__.__name__} #{self.id_}({self.latitude}, ' \
            f'{self.longitude})'

    __str__ = __repr__

    @property
    def coords(self) -> Coords:
        return self.latitude, self.longitude

    @property
    def map_coords(self) -> Coords:
        return xy(self.latitude, self.longitude % 360.0)

    def quandrant_bearing(self, to: 'IndexPoint') -> constants.Quadrant:
        if to.latitude >= self.latitude:
            if to.longitude >= self.longitude:
                return constants.Quadrant.Q_I
            else:
                return constants.Quadrant.Q_II
        else:
            if to.longitude >= self.longitude:
                return constants.Quadrant.Q_IV
            else:
                return constants.Quadrant.Q_III


class Point(IndexPoint):
    duplicates: t.List['Point']
    depth: int

    def __init__(self, id_: int, latitude: float, longitude: float):
        self.duplicates = []
        self.depth = -1
        super().__init__(id_, latitude, longitude)

    def merge_duplicates(self, duplicates: t.List['Point']) -> 'Point':
        if duplicates:
            self.duplicates.extend(duplicates)
        return self


Distance = t.Tuple[Point, float]


class Index:
    contents: t.List[Indexable]
    precision: int
    index: t.Optional[GeoGridIndex]
    indexed: bool

    def __init__(
        self,
        contents: t.List[Indexable],
        precision: int = constants.DEFAULT_PRECISION
    ):
        self.set_precision(precision)
        self.set(contents)

    def set_precision(self, precision):
        if precision not in GEO_HASH_GRID_SIZE:
            self.precision = constants.MIN_PRECISION
        else:
            self.precision = precision
        self.indexed = False
        self.index = None

    def append(self, *entries: Indexable):
        self.contents.extend(entries)
        self.indexed = False
        self.index = None

    def _search_radius(self) -> float:
        return GEO_HASH_GRID_SIZE[self.precision] / 2.0

    def set(self, contents: t.List[t.Any]):
        self.contents = contents
        self.index = None
        self.indexed = False

    def remove(self, entry: t.Any):
        self.set(list(filter(lambda c: c == entry, self.contents)))

    def build_index(self):
        if not self.indexed:
            self.index = GeoGridIndex(precision=self.precision)
            for c in self.contents:
                if isinstance(c, IndexPoint):
                    self.index.add_point(c)
            self.indexed = True

    def get_nearest(
        self, target: Point,
        resize: bool = True,
        min_count: int = constants.MIN_RESULT_COUNT,
    ) -> t.List[Distance]:
        while True:
            self.build_index()
            try:
                points = sorted(
                    filter(itemgetter(1),
                           self.index.get_nearest_points(
                               target, self._search_radius())),
                    key=itemgetter(1))
            except ValueError:
                points = []
            if not resize:
                return points
            elif (len(points) >= min_count or
                  self.precision == constants.MIN_PRECISION):
                return points
            else:
                self.set_precision(self.precision - 1)


Segment = t.List[Coords]


class Grid(IndexPoint, Index):
    radius: float
    depth: int
    graph: nx.Graph

    def __init__(
        self,
        latitude: float,
        longitude: float,
        contents: t.List[Indexable],
        radius: float = constants.INITIAL_RADIUS,
        depth: int = 0,
        id_: t.Optional[int] = None
    ):
        self.seed = None
        self.radius = radius
        self.depth = depth
        super().__init__(id_ or NUMBERS.next(), latitude, longitude)
        self.set(contents)
        self.set_precision(constants.DEFAULT_PRECISION)
        self.graph = nx.Graph(depth=self.depth)

    def __repr__(self):
        return (f'{self.__class__.__name__} #{self.id_}[{self.depth}]'
                f'({self.latitude}, {self.longitude})')

    __str__ = __repr__

    def __hash__(self) -> int:
        return self.id_

    def __eq__(self, other):
        if isinstance(other, Grid):
            return self.id_ == other.id_
        return False

    @property
    def is_ending(self) -> bool:
        for e in self.contents:
            if not isinstance(e, Grid):
                return True
        return False

    def sub_quadrants(self) -> (t.Tuple[float, t.Tuple[Coords,
                                                       Coords,
                                                       Coords,
                                                       Coords]]):
        new_radius = self.radius / 2.0
        return (new_radius,
                ((self.latitude + new_radius, self.longitude + new_radius),
                 (self.latitude + new_radius, self.longitude - new_radius),
                 (self.latitude - new_radius, self.longitude - new_radius),
                 (self.latitude - new_radius, self.longitude + new_radius)))

    def bounds(
        self,
    ) -> t.List[Coords]:
        lon, lat = self.map_coords
        lon1, lat1 = lon - self.radius, lat + self.radius
        lon2, lat2 = lon + self.radius, lat - self.radius
        if lon1 < 0.0 and lon2 == 0.0:
            lon2 = -0.00000000001
        return [(lon1, lat1),
                (lon2, lat1),
                (lon2, lat2),
                (lon1, lat2)]

    def redistribute(
        self,
        point: IndexPoint,
        quadrants: t.Tuple[Coords, Coords, Coords, Coords],
    ):
        bearing = self.quandrant_bearing(point)
        return (quadrants[bearing], bearing), point

    def subdivide(self):
        point_count = len(self.contents)
        if point_count <= constants.MAX_GRID_DENSITY:
            for entry in self.contents:
                entry.depth = self.depth
            if point_count == 1:
                self.graph.add_node(self.contents[0])
            elif point_count == 2:
                p1, p2 = self.contents
                self.graph.add_edge(p1, p2, weight=p1.distance_to(p2))
            return
        new_radius, quadrant_coords = self.sub_quadrants()
        new_depth = self.depth + 1
        redistributed_points = (
            self.redistribute(p, quadrant_coords)
            for p in self.contents if isinstance(p, IndexPoint))
        grouped_points = itertools.groupby(sorted(redistributed_points,
                                                  key=itemgetter(0)),
                                           key=itemgetter(0))
        new_contents = [Grid(lat, lon, list(map(itemgetter(1), points)),
                             radius=new_radius, depth=new_depth,
                             id_=(self.id_ * 10) + bearing)
                        for ((lat, lon), bearing), points in grouped_points]
        for c in new_contents:
            if isinstance(c, Grid):
                c.subdivide()
        self.set(new_contents)

    def terminals(self, child: bool = False) -> t.Iterable['Grid']:
        grids: t.List[Grid] = list(filter(lambda g: isinstance(g, Grid),
                                          self.contents))
        terminal_grids = list(filter(lambda g: g.is_ending, grids))
        local = []
        if not child and self.is_ending:
            local = [self]
        return itertools.chain(local, terminal_grids,
                               itertools.chain.from_iterable(
                                   (c.terminals(child=True)
                                    for c in grids)))

    def _get_graphs(self, child: bool = False) -> t.Iterable[nx.Graph]:
        grids = list(self.terminals(child=child))
        return itertools.chain(
            (self.graph,),
            itertools.chain.from_iterable((c._get_graphs(child=True)
                                           for c in grids))
        )

    def join_graphs(self) -> nx.Graph:
        new_graph = nx.Graph()
        all_graphs = self._get_graphs()
        for g in all_graphs:
            new_graph.add_edges_from(g.edges(data=True),
                                     depth=g.graph.get('depth'))
            new_graph.add_nodes_from(g.nodes(data=True),
                                     depth=g.graph.get('depth'))
        return new_graph

    def find(self, func: t.Callable[[t.Any], bool]) -> t.Any:
        return next(self.filter(func), None)

    def filter(
        self,
        func: t.Callable[[t.Any], bool]
    ) -> t.Iterator[t.Any]:
        return filter(func, self.contents)

    def __contains__(self, item):
        return self.find(lambda c: c == item) is not None

    @property
    def empty(self):
        return len(self.contents) == 0

    def zoom(self):
        lon, lat = self.map_coords
        lon1, lat1 = lon - self.radius, lat - self.radius
        lon2, lat2 = lon + self.radius, lat + self.radius
        return (lon, lat), (lon1, lat1), (lon2, lat2)

    def map(self) -> t.Tuple[t.Tuple[t.List[Coords], float], t.List[Coords],
                             t.List[Segment]]:
        return (
            (self.bounds(), self.radius),
            [n.map_coords for n in self.graph.nodes()],
            [[a.map_coords, b.map_coords] for a, b in list(self.graph.edges())],
        )


class DataSet:
    name: str
    edge_weight_type: EdgeWeightType
    grids: t.List[Grid]
    file_name: str

    @util.timeit
    def __init__(self, path_name):
        with open(path_name) as fh:
            self.file_name = path.basename(path_name)
            meta_data = self._read_metadata(fh)
            self.name = meta_data.get("name")
            self.edge_weight_type = meta_data.get("edge_weight_type")
            points = self._read_points(fh, self.edge_weight_type)
            self.grids = self._generate_grids(points)
        logger.info("Loaded %s points", len(points))
        logger.info("Generated %s grids", len(self.grids))

    @staticmethod
    def _read_metadata(fh: t.TextIO) -> t.Mapping[str, str]:
        meta_data = {}
        for read_line in fh:
            if COORD_DELIMITER in read_line:
                break
            field, value = read_line.strip().split(" : ")
            meta_data[field.lower()] = value
        return meta_data

    @staticmethod
    def _read_points(
        fh: t.TextIO,
        edge_weight_type: EdgeWeightType
    ) -> t.List[IndexPoint]:
        coord_parser = float
        if edge_weight_type == EdgeWeightType.EUC_2D:
            coord_parser = _euc_2d_parser
        coordinates: t.Dict[Coords, t.List[Point]] = {}
        for read_line in fh:
            try:
                id_, lat, lon = read_line.strip().split(" ")
                point = Point(int(id_), coord_parser(lat), coord_parser(lon))
                points = coordinates.setdefault(point.coords, [])
                points.append(point)
            except ValueError:
                pass
        return [first.merge_duplicates(rest)
                for first, *rest in coordinates.values()]

    @staticmethod
    def _generate_grids(points: t.Iterable[IndexPoint]) -> t.List[Grid]:
        points = ((p, _initial_grid_coords(p.latitude, p.longitude))
                  for p in points)
        grouped_points = itertools.groupby(sorted(points, key=itemgetter(1)),
                                           key=itemgetter(1))
        return [Grid(lat, lon, list(map(itemgetter(0), points)))
                for (lat, lon), points in grouped_points]


if __name__ == "__main__":
    target_path = util.get_relative_path(__file__, "../data/ar9152.tsp")
    logger.info("Loading %s", target_path)
    data_set = DataSet(target_path)
    input("")
