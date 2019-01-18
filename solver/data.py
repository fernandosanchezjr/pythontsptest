import enum
import itertools
import logging
import math
import typing as t
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


def _grid_coordinates(lat: float, lon: float) -> Coords:
    return (math.trunc(lat) + (-0.5 if lat < 0.0 else 0.5),
            math.trunc(lon) + (-0.5 if lon < 0.0 else 0.5))


def _euc_2d_parser(coord: str) -> float:
    return float(coord) * -0.001


def xy(latitude: float, longitude: float) -> Coords:
    return longitude, latitude


class IndexEntry(GeoPoint):
    numbers = util.Numbers()
    id_: int

    def __init__(self, id_: int, latitude: float, longitude: float):
        self.id_ = id_
        super().__init__(latitude, longitude)

    def __hash__(self) -> int:
        return self.id_

    def __repr__(self):
        """
        Machine representation of Grid instance.
        """
        return f'{self.__class__.__name__} #{self.id_}({self.latitude}, ' \
            f'{self.longitude})'

    __str__ = __repr__

    def map_coords(self) -> Coords:
        return xy(self.latitude, self.longitude)


Distance = t.Tuple[IndexEntry, float]
Line = t.Tuple[float, float, float, float]


class Point(IndexEntry):
    duplicates: t.List['Point']
    grid: Coords

    def __init__(self, id_: int, latitude: float, longitude: float):
        self.grid = _grid_coordinates(latitude, longitude)
        self.duplicates = []
        super().__init__(id_, latitude, longitude)

    @property
    def coords(self) -> Coords:
        return self.longitude, self.latitude

    def merge_duplicates(self, duplicates: t.List['Point']) -> 'Point':
        if duplicates:
            self.duplicates.extend(duplicates)
        return self


class Grid(IndexEntry):
    contents: t.List[IndexEntry]
    precision: int
    index: GeoGridIndex

    def __init__(
        self,
        latitude: float,
        longitude: float,
        contents: t.List[IndexEntry],
        precision: int = constants.DEFAULT_PRECISION
    ):
        self.contents = contents
        super().__init__(self.numbers.next(), latitude, longitude)
        self.precision = precision
        self._set_precision(precision)

    def _set_precision(self, precision):
        if self.precision == constants.MIN_PRECISION:
            return
        if precision not in GEO_HASH_GRID_SIZE:
            self.precision = constants.MIN_PRECISION
        else:
            self.precision = precision
        self.index = GeoGridIndex(precision=self.precision)
        for p in self.contents:
            self.index.add_point(p)

    def _radius(self) -> float:
        return GEO_HASH_GRID_SIZE[self.precision] / 2.0

    def get_nearest_points(
        self, target: GeoPoint,
        resize: bool = True
    ) -> t.List[Distance]:
        while True:
            points = sorted(
                filter(lambda n: n[1] > 0.0,
                       self.index.get_nearest_points(target, self._radius())),
                key=lambda n: n[1])
            if not resize:
                return points
            elif (len(points) >= constants.MIN_RESULT_COUNT or
                  self.precision == constants.MIN_PRECISION):
                return points
            else:
                self._set_precision(self.precision - 1)

    def map_coords(self) -> Coords:
        return xy(self.latitude, self.longitude)

    def bounds(
        self,
    ) -> t.List[Coords]:
        center_x, center_y = self.map_coords()
        x1, y1 = center_x - 0.5, center_y + 0.5
        x2, y2 = center_x + 0.5, center_y - 0.5
        if x1 < 0.0 and x2 == 0.0:
            x2 = 359.999999
        return [(x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
                (x1, y1)]


class DataSet:
    name: str
    edge_weight_type: EdgeWeightType
    points: t.List[Point]
    grids: t.List[Grid]
    file_name: str

    @util.timeit
    def __init__(self, path_name):
        with open(path_name) as fh:
            self.file_name = path.basename(path_name)
            meta_data = self._read_metadata(fh)
            self.name = meta_data.get("name")
            self.edge_weight_type = meta_data.get("edge_weight_type")
            self.points = self._read_points(fh, self.edge_weight_type)
            self.grids = self._generate_grids()
        logger.info("Loaded %s points", len(self.points))
        logger.info("Generated %s grids", len(self.grids))

    @staticmethod
    def _read_metadata(fh: t.TextIO) -> t.Mapping[str, str]:
        meta_data = {}
        for line in fh:
            if COORD_DELIMITER in line:
                break
            field, value = line.strip().split(" : ")
            meta_data[field.lower()] = value
        return meta_data

    @staticmethod
    def _read_points(
        fh: t.TextIO,
        edge_weight_type: EdgeWeightType
    ) -> t.List[Distance]:
        coord_parser = float
        if edge_weight_type == EdgeWeightType.EUC_2D:
            coord_parser = _euc_2d_parser
        coordinates: t.Dict[Coords, t.List[Point]] = {}
        for line in fh:
            try:
                id_, lat, lon = line.strip().split(" ")
                point = Point(int(id_), coord_parser(lat), coord_parser(lon))
                points = coordinates.setdefault(point.coords, [])
                points.append(point)
            except ValueError:
                pass
        return [first.merge_duplicates(rest)
                for first, *rest in coordinates.values()]

    def _generate_grids(self) -> t.List[Grid]:
        return [Grid(lat, lon, list(points))
                for (lat, lon), points in
                itertools.groupby(sorted(self.points, key=lambda p: p.grid),
                                  key=lambda p: p.grid)]


if __name__ == "__main__":
    target_path = util.get_relative_path(__file__, "../data/ar9152.tsp")
    logger.info("Loading %s", target_path)
    data_set = DataSet(target_path)
    input("")
