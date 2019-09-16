import numpy as np

T = np.array([[0, -1], [1, 0]])


def line_intersect(a1, a2, b1, b2):
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1


if __name__ == '__main__':
    pa1 = np.array([[2.0, 2.0], [2.0, 2.0]])
    pa2 = np.array([[0.0, 2.0], [-1.0, 2.0]])
    pb1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    pb2 = np.array([[1.0, 3.0], [1.5, 3.0]])

    intersects = line_intersect(pa1, pa2, pb1, pb2)
    print('intersects', intersects)
