import asyncio
import logging
import typing as t
from concurrent import futures

import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from new_solver import args, data, util

logger = logging.getLogger(__name__)


async def _processor(executor, queue, func: (), args_list: t.List[t.Any]):
    for f in executor.map(func, args_list):
        await queue.put(f)


async def _consumer(queue, chunk_count: int):
    completed_chunks = 0
    results = [None] * chunk_count
    while completed_chunks < chunk_count:
        chunk = await queue.get()
        results[completed_chunks] = chunk
        completed_chunks += 1
    return results


class Processor:
    _executor_count: int
    _executor: futures.ProcessPoolExecutor
    _loop: asyncio.events.AbstractEventLoop
    _queue: asyncio.Queue

    def __init__(self):
        self._executor_count = psutil.cpu_count()
        p = psutil.Process()
        p.cpu_affinity(range(psutil.cpu_count()))
        self._executor = futures.ProcessPoolExecutor(
            max_workers=self._executor_count
        )
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self.index = None

    def process(self, func: (), args_list: t.List[t.Any]):
        _, results = self._loop.run_until_complete(asyncio.gather(
            _processor(self._executor, self._queue, func, args_list),
            _consumer(self._queue, len(args_list))
        ))
        return results


def _do_test(grid: data.Grid):
    return grid


def test(processor: Processor, grids: t.List[data.Grid]) -> t.List[data.Grid]:
    new_grids = processor.process(_do_test, grids)
    return new_grids


def _do_convex_hull(grid: data.Grid):
    return grid.calculate_hull()


def convex_hulls(processor: Processor, grids: t.List[data.Grid]) -> t.List[data.Grid]:
    return processor.process(_do_convex_hull, grids)


@util.timeit
def process_grids(grids: t.List[data.Grid]) -> t.List[data.Grid]:
    proc = Processor()
    return convex_hulls(proc, grids)


@util.timeit
def draw_graph(name: str, grids: t.List[data.Grid]):
    all_points = np.concatenate([g.array() for g in grids])
    plt.figure(name)
    plt.scatter(x=all_points[:, 0], y=all_points[:, 1])
    patches = PatchCollection([Polygon(g.hull) for g in grids], edgecolors='red', facecolors='none')
    ax = plt.gca()
    ax.add_collection(patches)
    plt.draw()


@util.timeit
def main(show_map: bool = False):
    startup_args = args.parse_args()
    logger.info("Loading %s", startup_args.datafile)
    name, grids = data.load_datafile(startup_args.datafile)
    grids = process_grids(grids)
    if show_map:
        draw_graph(name, grids)
        plt.show()


if __name__ == "__main__":
    main(True)
