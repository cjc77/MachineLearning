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
            for N2 in [1, 5, 100]:
                A = np.triu(np.random.normal(size=(N + N2, N)))
                x = np.random.normal(size=(N,))
                b = A @ x
                x2 = regression.solve_triangular(A, b)
                assert_allclose(x, x2, atol=1e-5)
