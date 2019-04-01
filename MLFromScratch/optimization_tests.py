import regression
import numpy as np
from unittest import TestCase
from numpy.testing import assert_allclose
from scipy.stats import ortho_group
from numpy.linalg import det


class QRTestCase(TestCase):

    def test_2d(self):
        A = np.array([[1, 1], [0, 1]])
        b = np.array([2, 3])
        x = regression.solve_triangular(A, b)
        assert_allclose(x, np.array([-1, 3]))

    def test_solve_triangular(self):
        for N in range(1, 20):
            A = np.triu(np.random.normal(size=(N, N)))
            x = np.random.normal(size=(N,))
            b = A @ x
            x2 = regression.solve_triangular(A, b)
            assert_allclose(x, x2, atol=1e-5)

    def test_solve_rect_triangular(self):
        for N in range(1, 20):
            for N2 in [1, 5, 100]:
                A = np.triu(np.random.normal(size=(N + N2, N)))
                x = np.random.normal(size=(N,))
                b = A @ x
                x2 = regression.solve_triangular(A, b)
                assert_allclose(x, x2, atol=1e-5)

    def test_reflection(self):
        x = np.array([1, 1, 1])
        e1 = np.array([1, 0, 0])
        H = regression.householder_reflection(x, e1)
        assert_allclose(H @ (np.sqrt(3) * np.array([1, 0, 0])), x, atol=1e-5)
        assert_allclose(H @ np.array([1, 1, 1]), np.sqrt(3) * e1, atol=1e-5)

    def test_square_qr(self):
        A = np.array([[2, 1], [0, 3]])
        Q, R = regression.qr_decomposition(A)
        assert_allclose(Q, np.identity(2))
        assert_allclose(R, A)

        N = 3
        # generates random orthogonal matrices
        Q = ortho_group.rvs(N)
        R = np.triu(np.random.normal(size=(N, N)))
        A = Q @ R
        Q2, R2 = regression.qr_decomposition(Q @ R)

        assert_allclose(Q2 @ R2, Q @ R, atol=1e-5)
        assert_allclose(np.abs(det(Q2)), 1.0, atol=1e-5)
        assert_allclose(R2[2, 0], 0, atol=1e-5)
        assert_allclose(R2[2, 1], 0, atol=1e-5)
        assert_allclose(R2[1, 0], 0, atol=1e-5)

    def test_rect_qr(self):
        A = np.array([
            [2, 1],
            [0, 3],
            [4, 5],
            [1, 1],
        ])
        Q, R = regression.qr_decomposition(A)
        assert_allclose(R[1:, 0], np.zeros(A.shape[0] - 1), atol=1e-5)
        assert_allclose(R[2:, 0], np.zeros(A.shape[0] - 2), atol=1e-5)
        assert_allclose(Q @ R, A, atol=1e-5)
