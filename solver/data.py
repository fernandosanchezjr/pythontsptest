import enum
import itertools
import logging
import math
import typing as t
from operator import itemgetter
from os import path

from geoindex import GeoGridIndex, GeoPoint
from geoindex.geo_grid_index import GEO_HASH_GRID_SIZE

from solver import constants, util

logger = logging.getLogger(__name__)
COORD_DELIMITER = "NODE_COORD_SECTION"


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
    numbers: t.ClassVar[util.Numbers] = util.Numbers()
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

    def segment_to(self, to: 'Point', distance: float) -> 'Segment':
        return Segment(self, to, distance)


class Segment:
    id_: int
    raw_endpoints: t.Set[IndexPoint]
    distance: float
    depth: int

    class Pointer(Point):
        segment: 'Segment'

        # noinspection PyMissingConstructor
        def __init__(self, segment: 'Segment', point: Point):
            self.id_ = point.id_
            self.duplicates = point.duplicates
            self.latitude = point.latitude
            self.longitude = point.longitude
            self._rad_latitude = point._rad_latitude
            self._rad_longitude = point._rad_longitude
            self.depth = point.depth
            self.segment = segment

        def segment_to(self, to: Point, distance: float) -> 'Segment':
            return Segment(self, to, distance)

        def __hash__(self) -> int:
            return self.id_

        def __eq__(self, other):
            if isinstance(other, Segment):
                return self in other.raw_endpoints
            if isinstance(other, IndexPoint):
                return self.id_ == other.id_
            return False

    def __init__(
        self,
        entry1: Point,
        entry2: Point,
        distance: float
    ):
        self.id_ = IndexPoint.numbers.next()
        self.raw_endpoints = frozenset((Segment.Pointer(self, entry1),
                                        Segment.Pointer(self, entry2)))
        self.distance = distance

    def __hash__(self) -> int:
        return self.id_

    def __eq__(self, other) -> bool:
        if isinstance(other, Segment):
            return self.raw_endpoints == other.raw_endpoints
        elif isinstance(other, IndexPoint):
            return other in self.raw_endpoints
        return False

    @property
    def endpoints(self) -> t.Tuple[IndexPoint, IndexPoint]:
        a, b = self.raw_endpoints
        return a, b

    def __repr__(self):
        return f'{self.__class__.__name__} #{self.id_}(' \
            f'{self.endpoints[0]} -> {self.endpoints[1]} ' \
            f'distance={self.distance})'

    __str__ = __repr__

    @property
    def map_endpoints(self) -> t.List[Coords]:
        a, b = self.endpoints
        return [a.map_coords, b.map_coords]


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
                if isinstance(c, Segment):
                    for e in c.endpoints:
                        self.index.add_point(e)
                if isinstance(c, IndexPoint):
                    self.index.add_point(c)
            self.indexed = True

    @staticmethod
    def _deduplicate_segments(nearest: t.List[Distance]) -> t.List[Distance]:
        dupes = set()
        results = []
        for entry, distance in nearest:
            if isinstance(entry, Segment.Pointer):
                if entry.segment.id_ in dupes:
                    continue
                dupes.add(entry.segment.id_)
            results.append((entry, distance))
        return results

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
                return self._deduplicate_segments(points)
            elif (len(points) >= min_count or
                  self.precision == constants.MIN_PRECISION):
                return self._deduplicate_segments(points)
            else:
                self.set_precision(self.precision - 1)


class Cluster:
    segments: t.Set[Segment]
    points: t.Set[IndexPoint]

    def __init__(self):
        self.segments = set()
        self.points = set()

    def empty(self):
        return len(self.points) == 0

    def append(self, item):
        if isinstance(item, Segment):
            self.segments.add(item)
            for p in item.endpoints:
                self.points.add(p)
        elif isinstance(item, Cluster):
            for s in item.segments:
                self.segments.add(s)
            for p in item.points:
                self.points.add(p)
        elif isinstance(item, IndexPoint):
            self.points.add(item)

    def __contains__(self, item) -> bool:
        if isinstance(item, Segment):
            return item in self.segments
        elif isinstance(item, Cluster):
            return not bool(item.points - self.points)
        elif isinstance(item, IndexPoint):
            return item in self.points
        return False

    def __eq__(self, other):
        if isinstance(other, Cluster):
            return self.points == other.points
        return False

    def __gt__(self, other):
        if isinstance(other, Cluster):
            return len(self.points) > len(other.points)
        return False

    def __lt__(self, other):
        if isinstance(other, Cluster):
            return len(self.points) < len(other.points)
        return False

    def __repr__(self):
        return f'{self.__class__.__name__} ({len(self.points)})'

    __str__ = __repr__

    def intersects(self, item) -> bool:
        if isinstance(item, Segment):
            return bool(self.points & set(item.endpoints))
        elif isinstance(item, Cluster):
            return bool(self.points & item.points)
        elif isinstance(item, IndexPoint):
            return item in self.points
        return False


class Grid(IndexPoint, Index):
    radius: float
    depth: int
    seed: t.Optional[Point]

    def __init__(
        self,
        latitude: float,
        longitude: float,
        contents: t.List[Indexable],
        radius: float = constants.INITIAL_RADIUS,
        depth: int = 0
    ):
        self.seed = None
        self.radius = radius
        self.depth = depth
        super().__init__(self.numbers.next(), latitude, longitude)
        self.set(contents)
        self.set_precision(constants.DEFAULT_PRECISION)

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

    def subdivide(self):
        point_count = len(self.contents)
        if point_count <= constants.MAX_GRID_DENSITY:
            for entry in self.contents:
                entry.depth = self.depth
            if point_count == 1:
                self.seed = self.contents[0]
            elif point_count == 2:
                p1, p2 = self.contents
                segment = p1.segment_to(p2, p1.distance_to(p2))
                self.set([segment])
            return
        new_radius, quadrant_coords = self.sub_quadrants()
        new_depth = self.depth + 1
        redistributed_points = ((c, quadrant_coords[self.quandrant_bearing(c)])
                                for c in self.contents if isinstance(c, IndexPoint))
        grouped_points = itertools.groupby(sorted(redistributed_points,
                                                  key=itemgetter(1)),
                                           key=itemgetter(1))
        new_contents = [Grid(lat, lon, list(map(itemgetter(0), points)),
                             radius=new_radius, depth=new_depth)
                        for (lat, lon), points in grouped_points]
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
        return itertools.chain(local, terminal_grids, itertools.chain.from_iterable((c.terminals(child=True)
                                                                                     for c in grids)))

    def endpoints(self, child: bool = False) -> t.Iterable[t.Any]:
        grids = list(self.terminals(child=child))
        return itertools.chain(
            filter(lambda c: isinstance(c, (Point, Segment)), self.contents),
            itertools.chain.from_iterable((c.endpoints(child=True) for c in grids)))

    def find(self, func: t.Callable[[t.Any], bool]) -> t.Any:
        return next(filter(func, self.contents), None)

    def filter(
        self,
        func: t.Callable[[t.Any], bool]
    ) -> t.Iterator[t.Any]:
        return filter(func, self.contents)

    def __contains__(self, item):
        return self.find(lambda c: c == item) is not None

    def sieve(
        self,
        entry: IndexPoint
    ) -> t.Tuple[t.Optional[IndexPoint], t.List['Grid']]:
        if entry in self:
            return entry, [self]
        _, quadrant_coords = self.sub_quadrants()
        coords = quadrant_coords[self.quandrant_bearing(entry)]
        next_grid: t.Optional[Grid] = next(self.filter(
            lambda ng: isinstance(ng, Grid) and ng.coords == coords), None)
        if not next_grid:
            return None, [self]
        target, route = next_grid.sieve(entry)
        return target, [self] + route

    @property
    def empty(self):
        return len(self.contents) == 0

    def zoom(self):
        lon, lat = self.map_coords
        lon1, lat1 = lon - self.radius, lat - self.radius
        lon2, lat2 = lon + self.radius, lat + self.radius
        return (lon, lat), (lon1, lat1), (lon2, lat2)


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


def common_path(*paths: t.List[Grid]) -> t.Tuple[t.List[Grid],
                                                 t.List[t.List[Grid]]]:
    prefix = list(map(itemgetter(0),
                      itertools.takewhile(lambda x: all(x[0] == y for y in x),
                                          zip(*paths))))
    remainders = [p[len(prefix):] for p in paths]
    return prefix, remainders


def remove_nested_entry(
    route: t.List[Grid],
    entry: IndexPoint
) -> t.List[Grid]:
    reversed_path = reversed(route)
    new_route = []
    for grid in reversed_path:
        if entry in grid:
            grid.remove(entry)
        if grid.empty:
            entry = grid
        else:
            new_route.append(grid)
    return list(reversed(new_route))


if __name__ == "__main__":
    target_path = util.get_relative_path(__file__, "../data/ar9152.tsp")
    logger.info("Loading %s", target_path)
    data_set = DataSet(target_path)
    input("")
