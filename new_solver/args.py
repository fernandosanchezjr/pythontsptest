import argparse

from new_solver import constants, util

DEFAULT_DATA_FILE = util.get_relative_path(__file__, constants.DATA_FILE)


def parse_args():
    parser = argparse.ArgumentParser("Research TSP Solver")
    parser.add_argument("--datafile", type=str, default=DEFAULT_DATA_FILE)
    parsed, _ = parser.parse_known_args()
    return parsed
