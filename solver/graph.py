import logging
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.basemap import Basemap

from solver import data, util

logger = logging.getLogger(__name__)

GridCoords = t.List[PolyCollection]
PointCoords = t.Optional[t.Tuple[t.Any, t.Any]]
SegmentCoords = t.Optional[LineCollection]
MapData = t.Tuple[GridCoords, PointCoords, SegmentCoords]


class Map:
    numbers = util.Numbers()
    title: t.Any

    def __init__(self, title=""):
        self.title = title
        self.fig = plt.figure(self.title or self.numbers.next())
        # miller projection
        self.world_map = Basemap(projection='mill', lon_0=180)
        # plot coastlines, draw label meridians and parallels.
        self.world_map.drawcoastlines()
        self.world_map.drawparallels(
            np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
        self.world_map.drawmeridians(
            np.arange(self.world_map.lonmin, self.world_map.lonmax + 30, 60),
            labels=[0, 0, 0, 1])
        # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
        self.world_map.drawmapboundary(fill_color='aqua')
        self.world_map.fillcontinents(color='coral', lake_color='aqua')
        if title:
            plt.title(self.title)

    def to_map_xy(self, entries: t.List[data.Coords]) -> t.Tuple[t.Any, t.Any]:
        bounds = np.array(entries)
        x, y = bounds.T
        return self.world_map(x, y)

    @staticmethod
    def plot_grids(grids: GridCoords):
        if grids:
            gca = plt.gca()
            for grid in grids:
                gca.add_collection(grid)

    @staticmethod
    def plot_points(points: PointCoords, color='black',
                    markersize=0.8):
        if points:
            x, y = points
            plt.plot(x, y, 'ok', markersize=markersize, color=color,
                     zorder=3.0)

    @staticmethod
    def plot_segments(segments: t.Optional[LineCollection]):
        if segments:
            gca = plt.gca()
            gca.add_collection(segments)

    def grid_to_map(self, grid) -> PolyCollection:
        return PolyCollection(np.dstack(self.to_map_xy(grid.bounds())),
                              edgecolors='blue', facecolors='none',
                              linewidths=1.0 + grid.radius, zorder=2.0)

    def points_to_map(
        self,
        points: t.List[data.Point]
    ) -> t.Optional[t.Tuple[t.Any, t.Any]]:
        if not points:
            return None
        return self.to_map_xy([p.map_coords for p in points])

    def endpoint_to_map(
        self,
        endpoints: t.List[data.Coords]
    ) -> t.List[data.Coords]:
        return list(np.dstack(self.to_map_xy(endpoints)))

    def segments_to_map(
        self,
        segments: t.List[data.Segment]
    ) -> t.Optional[LineCollection]:
        if not segments:
            return None
        lines = []
        for s in segments:
            lines.extend(np.dstack(self.to_map_xy(s.map_endpoints)))
        return LineCollection(lines, colors='green', linewidths=0.75,
                              linestyles='solid', zorder=4)

    def generate_data(
        self,
        grid: data.Grid
    ) -> MapData:
        terminals = []
        points = []
        segments = []
        for terminal in grid.get_terminals():
            terminals.append(self.grid_to_map(terminal))
            for entry in terminal.contents:
                if isinstance(entry, data.Point):
                    points.append(entry)
                elif isinstance(entry, data.Segment):
                    segments.append(entry)
        return terminals, self.points_to_map(points), self.segments_to_map(
            segments)

    def draw_data(
        self,
        grids: GridCoords,
        points: PointCoords = None,
        segments: SegmentCoords = None
    ):
        plt.figure(self.fig.number)
        self.plot_grids(grids)
        self.plot_points(points)
        self.plot_segments(segments)

    def save(self, file_name="graph.eps", file_format="eps"):
        plt.figure(self.fig.number)
        self.fig.savefig(file_name, format=file_format)

    @classmethod
    def show(cls):
        plt.show()


if __name__ == "__main__":
    m = Map("test!")
    ps = m.points_to_map([data.Point(2360, -54.2666667, -66.7666667),
                          data.Point(6409, -54.45, -66.5)])
    m.plot_points(ps)
    m.save()
