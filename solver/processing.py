import itertools
import logging
import time
import typing as t

from geoindex import GeoGridIndex, GeoPoint
from geoindex.geo_grid_index import GEO_HASH_GRID_SIZE

from solver import data, util


logger = logging.getLogger(__name__)

MIN_GRID_SIZE = 1
MIN_RESULT_COUNT = 2


def group_points(points: t.List[data.Point]) -> t.List[GeoPoint]:
    return [GeoPoint(lat, lon, ref=list(g))
            for (lat, lon), g in
            itertools.groupby(sorted(points, key=lambda p: p.grid),
                              lambda p: p.grid)]


class Grid:
    points: t.List[GeoPoint]
    precision: int
    index: GeoGridIndex

    def __init__(self, points: t.List[GeoPoint], precision: int = 5):
        self.points = points
        self._set_precision(precision)

    def _set_precision(self, precision):
        self.precision = precision
        if self.precision not in GEO_HASH_GRID_SIZE:
            self.precision = MIN_GRID_SIZE
        self.index = GeoGridIndex(precision=self.precision)
        for p in self.points:
            self.index.add_point(p)

    def _get_radius(self) -> float:
        return GEO_HASH_GRID_SIZE[self.precision] / 2.0

    def get_nearest_points(self, p: GeoPoint,
                           resize: bool = True
                           ) -> t.List[t.Tuple[GeoPoint, float]]:
        while True:
            nearest = sorted(
                filter(lambda n: n[1] > 0.0,
                       self.index.get_nearest_points(p, self._get_radius())),
                key=lambda n: n[1])
            if not resize:
                return nearest
            elif len(nearest) >= MIN_RESULT_COUNT or \
                    self.precision == MIN_GRID_SIZE:
                return nearest
            else:
                self._set_precision(self.precision - 1)


def load_grid(target_path: str) -> Grid:
    return Grid(group_points(data.DataSet(target_path).points))


if __name__ == "__main__":
    file_name = util.get_relative_path(__file__, "../data/ar9152.tsp")
    start = time.time()
    grid = load_grid(file_name)
    ns = grid.get_nearest_points(grid.points[0])
    logging.info("Elapsed time: %f sec", time.time() - start)
    input("")
