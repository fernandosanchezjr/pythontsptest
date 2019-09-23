import typing as t

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from shapely.geometry import MultiLineString

from solver import data, util

matplotlib.use('TkAgg')  # <-- THIS MAKES IT FAST!


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
        self.draw_background()

    def draw_background(self):
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
        if self.title:
            plt.title(self.title)

    def _to_map_xy(self, entries: t.List[data.Coords]) -> np.ndarray:
        bounds = np.array(entries)
        x, y = bounds.T
        return self.world_map(*self.world_map.shiftdata(
            x, datain=y, lon_0=self.center[0], fix_wrap_around=True))

    def _lines_to_map_xy(
        self,
        entries: t.Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        x, y = entries
        new_x, new_y = self.world_map(
            *self.world_map.shiftdata(
                x, datain=y, lon_0=self.center[0], fix_wrap_around=True
            )
        )
        return np.array(list(zip(new_x, new_y)))

    def draw_lines(
        self,
        lines: MultiLineString,
        colors: str = 'blue',
        linestyles: str = 'solid',
        linewidths: float = 1.0,
        zorder: float = 1,
    ):
        map_lines = [Path(np.array(self._lines_to_map_xy(g.xy)))
                     for g in lines.geoms]
        if not map_lines:
            return
        drawn_lines = PathCollection(
            map_lines,
            linewidths=linewidths,
            edgecolors=colors,
            facecolors='none',
            linestyles=linestyles,
            zorder=zorder)
        plt.figure(self.fig.number)
        gca = plt.gca()
        gca.add_collection(drawn_lines)

    def draw_points(
        self,
        points: t.List[data.Coords],
        color: str = 'yellow',
        markersize: float = 1.0,
        zorder: float = 1,
    ):
        x, y = self._to_map_xy(points)
        plt.figure(self.fig.number)
        plt.plot(x, y, 'ok', markersize=markersize, color=color, zorder=zorder)

    def save(self, file_name="graph.eps", file_format="eps"):
        plt.figure(self.fig.number)
        self.fig.savefig(file_name, format=file_format)

    @classmethod
    def show(cls):
        plt.show()
