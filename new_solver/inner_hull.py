from shapely.geometry import LineString

from new_solver import util


@util.timeit
def test_intersection(l1, l2):
    l1.intersects(l2)


@util.timeit
def test_intersection2(l1, l2):
    for _ in range(1000):
        l1.intersects(l2)


if __name__ == '__main__':
    line1 = LineString([(1.0, 1.0), (1.0, 3.0)])
    line2 = LineString([(2.0, 2.0), (-1.0, 2.0)])
    line3 = LineString([(2.0, 2.0), (1.0, 5.0)])

    test_intersection(line1, line2)
    test_intersection2(line1, line2)

    print('intersects 1', line1.intersects(line2))
    print('intersects 2', line1.intersects(line3))
