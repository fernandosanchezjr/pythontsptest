import typing as t

import matplotlib

matplotlib.use('TkAgg')  # <-- THIS MAKES IT FAST!
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import (LineCollection, PolyCollection)
from mpl_toolkits.basemap import Basemap

from solver import data, util

GridCoords = t.List[PolyCollection]
PointCoords = t.Optional[t.Tuple[t.Any, t.Any]]
SegmentCoords = t.List[t.List[data.Segment]]
MapData = t.Tuple[GridCoords, PointCoords, SegmentCoords]

Grids = t.List[t.Tuple[t.List[data.Coords], float]]
Points = t.List[data.Coords]
Segments = t.List[data.Segment]


class Map:
    numbers = util.Numbers()
    title: t.Any
    center: data.Coords

    def __init__(
        self,
        title="",
        center: data.Coords = (0, 0),
        bottom_left: data.Coords = (-180, -90),
        top_right: data.Coords = (180, 90),
    ):
        self.title = title
        self.fig = plt.figure(self.title or self.numbers.next())
        self.center = center
        # miller projection
        lon_0, lat_0 = center
        llcrnrlon, llcrnrlat = bottom_left
        urcrnrlon, urcrnrlat = top_right
        self.world_map = Basemap(projection='mill', lon_0=lon_0, lat_0=lat_0,
                                 llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                                 urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)
        # plot coastlines, draw label meridians and parallels.
        self.world_map.drawcoastlines()
        self.world_map.drawparallels(np.arange(-90, 90, 30),
                                     labels=[1, 0, 0, 0])
        self.world_map.drawmeridians(np.arange(self.world_map.lonmin,
                                               self.world_map.lonmax + 30, 60),
                                     labels=[0, 0, 0, 1])
        # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
        self.world_map.drawmapboundary(fill_color='aqua')
        self.world_map.fillcontinents(color='coral', lake_color='aqua')
        if title:
            plt.title(self.title)

    def to_map_xy(self, entries: t.List[data.Coords]) -> t.Tuple[t.Any, t.Any]:
        bounds = np.array(entries)
        x, y = bounds.T
        return self.world_map(*self.world_map.shiftdata(
            x, datain=y, lon_0=self.center[0], fix_wrap_around=True))

    @staticmethod
    def plot_grids(grids: GridCoords):
        if grids:
            gca = plt.gca()
            for grid in grids:
                gca.add_collection(grid)

    @staticmethod
    def plot_points(points: PointCoords, color='black',
                    markersize=1.0):
        if points:
            x, y = points
            plt.plot(x, y, 'ok', markersize=markersize, color=color,
                     zorder=3.0)

    @staticmethod
    def plot_segments(segments: SegmentCoords, colors: str = 'green'):
        if segments:
            gca = plt.gca()
            gca.add_collection(LineCollection(segments, colors=colors,
                                              linewidths=0.75,
                                              linestyles='solid', zorder=4))

        return

    def grids_to_map(
        self,
        grids: Grids
    ) -> GridCoords:
        return [PolyCollection(np.dstack(self.to_map_xy(bounds)),
                               edgecolors='blue', facecolors='none',
                               linewidths=1.0 + radius, zorder=2.0)
                for bounds, radius in grids]

    def points_to_map(
        self,
        points: t.List[data.Coords]
    ) -> PointCoords:
        if not points:
            return None
        return self.to_map_xy(points)

    def segments_to_map(
        self,
        segments: Segments
    ) -> SegmentCoords:
        lines = []
        for s in segments:
            lines.extend(np.dstack(self.to_map_xy(s)))
        return lines

    def draw_grids(
        self,
        grids: Grids,
        points: Points = None,
        internal_segments: Segments = None,
        external_segments: Segments = None
    ):
        map_grids = self.grids_to_map(grids)
        map_points = self.points_to_map(points) if points else []
        map_segments = (self.segments_to_map(internal_segments)
                        if internal_segments else [])
        map_hull_segments = (self.segments_to_map(external_segments)
                             if external_segments else [])
        plt.figure(self.fig.number)
        self.plot_grids(map_grids)
        self.plot_points(map_points)
        self.plot_segments(map_hull_segments, colors='yellow')
        self.plot_segments(map_segments)

    def save(self, file_name="graph.eps", file_format="eps"):
        plt.figure(self.fig.number)
        self.fig.savefig(file_name, format=file_format)

    @classmethod
    def show(cls):
        plt.show()


if __name__ == "__main__":
    m = Map("test!")
    ps = m.points_to_map([
        data.Point(2360, -54.2666667, -66.7666667).map_coords,
        data.Point(6409, -54.45, -66.5).map_coords])
    m.plot_points(ps)
    m.save()
