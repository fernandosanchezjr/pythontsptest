import typing as t

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from solver import data, util


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

    def add_entries(self, entries: t.List[data.IndexEntry]):
        plt.figure(self.fig.number)
        coords = np.array([p.map_coords(self.world_map) for p in entries])
        x, y = coords.T
        plt.plot(x, y, 'ok', markersize=2, color='red')

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
    m.add_entries([data.Point(2360, -54.2666667, -66.7666667),
                   data.Point(6409, -54.45, -66.5)])
    m.save()
    m.draw()
    m.show()
