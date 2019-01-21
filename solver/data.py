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


class IndexEntry(GeoPoint):
    numbers: t.ClassVar[util.Numbers] = util.Numbers()
    id_: int

    def __init__(self, id_: int, latitude: float, longitude: float):
        self.id_ = id_
        super().__init__(latitude, longitude)

    def __hash__(self) -> int:
        return self.id_

    def __repr__(self):
        return f'{self.__class__.__name__} #{self.id_}({self.latitude}, ' \
            f'{self.longitude})'

    __str__ = __repr__

    @property
    def coords(self) -> Coords:
        return xy(self.latitude, self.longitude)

    @property
    def map_coords(self) -> Coords:
        return xy(self.latitude, self.longitude % 360.0)

    def quandrant_bearing(self, to: 'IndexEntry') -> constants.Quadrant:
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

    def segment(self, to: 'IndexEntry', distance: float) -> 'Segment':
        return Segment(self, to, distance)


class Segment:
    id_: int
    endpoints: t.Tuple[IndexEntry, IndexEntry]
    distance: float

    class Pointer(IndexEntry):
        segment: 'Segment'
        entry: IndexEntry

        def __init__(self, segment: 'Segment', entry: IndexEntry):
            self.__dict__['segment'] = segment
            self.__dict__['entry'] = entry
            super().__init__(entry.id_, entry.latitude, entry.longitude)

        def __getstate__(self):
            return self.__dict__

        def __setstate__(self, state):
            self.__dict__ = state

        def __getattr__(self, item):
            print("item:", item)
            return (getattr(self.segment, item, None) or
                    getattr(self.entry, item, None))

    def __init__(
        self,
        entry1: IndexEntry,
        entry2: IndexEntry,
        distance: float
    ):
        self.id_ = IndexEntry.numbers.next()
        self.endpoints = (Segment.Pointer(self, entry1),
                          Segment.Pointer(self, entry2))
        self.distance = distance

    def __hash__(self):
        return self.id_

    def __repr__(self):
        return f'{self.__class__.__name__} #{self.id_}(' \
            f'{self.endpoints[0]} -> {self.endpoints[1]} ' \
            f'distance={self.distance})'

    __str__ = __repr__


Distance = t.Tuple[IndexEntry, float]
Line = t.Tuple[float, float, float, float]


class Point(IndexEntry):
    duplicates: t.List['Point']

    def __init__(self, id_: int, latitude: float, longitude: float):
        self.duplicates = []
        super().__init__(id_, latitude, longitude)

    def merge_duplicates(self, duplicates: t.List['Point']) -> 'Point':
        if duplicates:
            self.duplicates.extend(duplicates)
        return self


Indexable = t.Union[IndexEntry, Segment]


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
        self.set_contents(contents)
        self.set_precision(precision)

    def set_precision(self, precision):
        if precision not in GEO_HASH_GRID_SIZE:
            self.precision = constants.MIN_PRECISION
        else:
            self.precision = precision
        self.indexed = False
        self.index = None

    def _search_radius(self) -> float:
        return GEO_HASH_GRID_SIZE[self.precision] / 2.0

    def set_contents(self, contents: t.List[Indexable]):
        self.contents = contents
        self.index = None
        self.indexed = False

    def build_index(self):
        if not self.indexed:
            self.index = GeoGridIndex(precision=self.precision)
            for c in self.contents:
                if isinstance(c, IndexEntry):
                    self.index.add_point(c)
                elif isinstance(c, Segment):
                    for e in c.endpoints:
                        self.index.add_point(e)
            self.indexed = True

    def get_nearest(
        self, target: IndexEntry,
        resize: bool = True
    ) -> t.List[Distance]:
        while True:
            self.build_index()
            try:
                points = sorted(
                    filter(lambda n: n[1] > 0.0,
                           self.index.get_nearest_points(
                               target, self._search_radius())),
                    key=itemgetter(1))
            except ValueError:
                points = []
            if not resize:
                return points
            elif (len(points) >= constants.MIN_RESULT_COUNT or
                  self.precision == constants.MIN_PRECISION):
                return points
            else:
                self.set_precision(self.precision - 1)


class Grid(IndexEntry, Index):
    radius: float
    subdivided: bool
    depth: int

    def __init__(
        self,
        latitude: float,
        longitude: float,
        contents: t.List[Indexable],
        radius: float = constants.INITIAL_RADIUS,
        depth: int = 0
    ):
        self.subdivided = False
        self.radius = radius
        self.depth = depth
        super().__init__(self.numbers.next(), latitude, longitude)
        self.set_contents(contents)
        self.set_precision(constants.DEFAULT_PRECISION)

    def __repr__(self):
        return f'{self.__class__.__name__} #{self.id_}[{self.depth}]' \
            f'({self.latitude}, {self.longitude})'

    __str__ = __repr__

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
        lat1, lon1 = self.latitude + self.radius, self.longitude - self.radius
        lat2, lon2 = self.latitude - self.radius, self.longitude + self.radius
        if lon1 < 0.0 and lon2 == 0.0:
            lon2 = -0.00000000001
        return [xy(lat1, lon1),
                xy(lat1, lon2),
                xy(lat2, lon2),
                xy(lat2, lon1),
                xy(lat1, lon1)]

    def subdivide(self):
        if self.subdivided:
            return
        point_count = len(self.contents)
        if point_count <= constants.MAX_GRID_DENSITY:
            if point_count == 2:
                p1, p2 = self.contents
                segment = Segment(p1, p2, p1.distance_to(p2))
                self.set_contents([segment])
                logger.info("Terminal %s: %s", self, self.contents)
            return
        new_radius, quadrant_coords = self.sub_quadrants()
        new_depth = self.depth + 1
        redistributed_points = ((c, quadrant_coords[self.quandrant_bearing(c)])
                                for c in self.contents)
        grouped_points = itertools.groupby(sorted(redistributed_points,
                                                  key=itemgetter(1)),
                                           key=itemgetter(1))
        new_contents = [Grid(lat, lon, list(map(itemgetter(0), points)),
                             radius=new_radius, depth=new_depth)
                        for (lat, lon), points in grouped_points]
        for c in new_contents:
            if isinstance(c, Grid):
                c.subdivide()
        self.set_contents(new_contents)
        self.subdivided = True

    def get_terminals(self):
        if not self.subdivided:
            return [self]
        return itertools.chain.from_iterable((c.get_terminals()
                                              for c in self.contents
                                              if isinstance(c, Grid)))

    def get_nearest(
        self, target: IndexEntry,
        resize: bool = True
    ) -> t.List[Distance]:
        return self.index.get_nearest(target, resize=resize)


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
    ) -> t.List[IndexEntry]:
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
    def _generate_grids(points: t.Iterable[IndexEntry]) -> t.List[Grid]:
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
