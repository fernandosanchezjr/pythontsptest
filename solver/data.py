# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:03:17 2018

@author: fernando
"""
import enum
import logging
import math
import time
import typing as t

from geoindex import GeoPoint

from solver import util

logger = logging.getLogger(__name__)
COORD_DELIMITER = "NODE_COORD_SECTION"


class EdgeWeightType(str, enum.Enum):
    EUC_2D = "EUC_2D"
    GEOM = "GEOM"


def _grid_coordinates(lat: float, lon: float) -> t.Tuple[float, float]:
    return math.trunc(lat) + 0.5, math.trunc(lon) + 0.5


def _euc_2d_parser(coord: str) -> float:
    return float(coord) * -0.001


class Point(GeoPoint):
    id_: int
    grid: t.Tuple[int, int]

    def __init__(self, id_, latitude, longitude):
        self.id_ = id_
        self.grid = _grid_coordinates(longitude, latitude)
        super().__init__(latitude, longitude)


class DataSet:
    name: str
    edge_weight_type: EdgeWeightType
    points: t.List[Point]

    def __init__(self, path_name):
        with open(path_name) as fh:
            meta_data = self._read_metadata(fh)
            self.name = meta_data.get("name")
            self.edge_weight_type = meta_data.get("edge_weight_type")
            self.points = self._read_points(fh, self.edge_weight_type)

    @staticmethod
    def _read_metadata(fh: t.TextIO) -> t.Mapping[str, str]:
        meta_data = {}
        for line in fh:
            current_line = line.strip()
            if current_line == COORD_DELIMITER:
                break
            field, value = current_line.split(" : ")
            meta_data[field.lower()] = value
        return meta_data

    @staticmethod
    def _read_points(fh: t.TextIO,
                     edge_weight_type: EdgeWeightType) -> t.List[Point]:
        points = []
        coord_parser = float
        if edge_weight_type == EdgeWeightType.EUC_2D:
            coord_parser = _euc_2d_parser
        for line in fh:
            try:
                id_, lon, lat = line.strip().split(" ")
                lat, lon = coord_parser(lat), coord_parser(lon)
                points.append(Point(int(id_), lat, lon))
            except ValueError:
                pass
        return points


if __name__ == "__main__":
    target_path = util.get_relative_path(__file__, "../data/world.tsp")
    logger.info("Loading %s", target_path)
    start = time.time()
    data_set = DataSet(target_path)
    logging.info("Elapsed time: %f sec", time.time() - start)
    logger.info("loaded %s points", len(data_set.points))
    input("")
