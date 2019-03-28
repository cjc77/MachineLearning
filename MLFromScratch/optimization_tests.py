import regression
import numpy as np
from unittest import TestCase
from numpy.testing import assert_allclose


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
