import numpy as np


def _split(u, v, points):
    # return points on left side of UV
    return [p for p in points if np.cross(p - u, v - u) < 0]


def _extend(u, v, points):
    if not points:
        return []

    # find furthest point W, and split search to WV, UW
    w = min(points, key=lambda p: np.cross(p - u, v - u))
    p1, p2 = _split(w, v, points), _split(u, w, points)
    return _extend(w, v, p1) + [w] + _extend(u, w, p2)


def convex_hull(points):
    # find two hull points, U, V, and split to left and right search
    u = min(points, key=lambda p: p[0])
    v = max(points, key=lambda p: p[0])
    left, right = _split(u, v, points), _split(v, u, points)

    # find convex hull on each side
    return [v] + _extend(u, v, left) + [u] + _extend(v, u, right) + [v]
