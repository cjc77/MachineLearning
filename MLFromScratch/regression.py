import numpy as np
from numpy.linalg import norm


def householder_reflection(a, e):
    u = a - np.sign(a[0]) * norm(a) * e
    v = u / norm(u)
    H = np.identity(len(a)) - 2 * np.outer(v, v)
    return H


def qr_decomposition(A):
    m, n = A.shape
    # A must have more rows than columns
    assert m >= n

    Q = np.identity(m)
    R = A.copy()

    # If matrix is square, can stop at m - 1
    square = int(m == n)
    for i in range(n - square):
        r = R[i:, i]
        # No need to zero out entries that are already zero
        if np.allclose(r[1:], 0.0):
            continue

        e = np.zeros(m - i)
        e[0] = 1
        H = np.identity(m)
        H[i:, i:] = householder_reflection(r, e)

        Q = Q @ H.T
        R = H @ R

    return Q, R


def solve_triangular(A, b):
    """Solve Ax = b when A is upper triangular & square"""

    m, n = A.shape
    x = b[(n - 1):n] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        back_subs = np.dot(A[i, (i + 1):], x)
        rhs = b[i] - back_subs
        x_i = rhs / A[i, i]
        x = np.insert(x, 0, x_i)

    return x