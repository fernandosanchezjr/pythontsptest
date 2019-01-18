import logging
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from solver import data, util

logger = logging.getLogger(__name__)


class Map:
    numbers = util.Numbers()

    def __init__(self, title=""):
        self.fig = plt.figure(title or self.numbers.next())
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
            plt.title(title)

    def to_xy(self, entries: t.List[t.Any]) -> t.Tuple[t.Any, t.Any]:
        bounds = np.array(entries)
        x, y = bounds.T
        return self.world_map(np.mod(x, 360.0), y)

    def add_points(self, points: t.List[data.IndexEntry], color='black',
                   markersize=0.8):
        if points:
            plt.figure(self.fig.number)
            x, y = self.to_xy([p.map_coords() for p in points])
            plt.plot(x, y, 'ok', markersize=markersize, color=color)

    def add_grids(self, grids: t.List[data.Grid]):
        if grids:
            plt.figure(self.fig.number)
            for grid in grids:
                x, y = self.to_xy(grid.bounds())
                plt.plot(x, y, color='blue', linestyle='-', linewidth=1.0)

    def save(self, file_name="graph.png"):
        plt.figure(self.fig.number)
        self.fig.savefig(file_name)

    def draw(self, block=False):
        plt.figure(self.fig.number)
        plt.draw()

    @classmethod
    def show(cls):
        plt.show()


if __name__ == "__main__":
    m = Map("test!")
    m.add_points([data.Point(2360, -54.2666667, -66.7666667),
                  data.Point(6409, -54.45, -66.5)])
    m.save()
    m.draw()
    m.show()
