from enum import Enum

MIN_PRECISION = 1
MIN_RESULT_COUNT = 2
DEFAULT_PRECISION = 4
MAX_GRID_DENSITY = 2
INITIAL_RADIUS = 0.5


class Quadrant(int, Enum):
    Q_I = 0
    Q_II = 1
    Q_III = 2
    Q_IV = 3
