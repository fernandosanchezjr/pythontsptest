from enum import Enum

DATA_FILE = "../data/ar9152.tsp"

MIN_PRECISION = 1
MIN_RESULT_COUNT = 2
DEFAULT_PRECISION = 4
MAX_GRID_DENSITY = 2
INITIAL_RADIUS = 0.5
SEED_DISTANCES = 10


class Quadrant(int, Enum):
    Q_I = 0
    Q_II = 1
    Q_III = 2
    Q_IV = 3


class EdgeWeightType(str, Enum):
    EUC_2D = "EUC_2D"
    GEOM = "GEOM"
