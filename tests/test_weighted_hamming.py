import numpy as np
from weighted_hamming import distance_matrix, distance_matrix_xy


class TestDistanceMatrix:
    def test_simple(self):
        X = np.array([[1,2],
                      [3,4]], dtype=float)
        w = np.array([5,6], dtype=float)

        D = distance_matrix(X, w)

        assert np.allclose(D, np.array([[0, 11], [11,0]]) / w.shape[0])

    def test_symmetry(self):
        X = np.random.random(size=(10,10))
        w = np.random.random(size=10)

        D = distance_matrix(X, w)

        assert np.allclose(D, D.T)
        assert np.abs(np.diag(D)).sum() == 0

    def test_large_x(self):
        # measure execution time with pytest
        X = np.random.randint(0, 100, size=(10000,500)).astype(float)
        w = np.random.random(size=500)

        D = distance_matrix(X, w)


class TestDistanceMatrixXY:
    def test_simple(self):
        X = np.array([[1, 2],
                      [3, 4]], dtype=float)
        Y = np.array([[1, 2],
                      [3, 4.01]], dtype=float)
        w = np.array([5, 6], dtype=float)

        D = distance_matrix_xy(X, Y, w)

        assert np.allclose(D, np.array([[0, 11], [11, 6]]) / w.shape[0])

    def test_large_xy(self):
        # measure execution time with pytest
        X = np.random.randint(0, 100, size=(10000, 500)).astype(float)
        Y = np.random.randint(0, 100, size=(5000, 500)).astype(float)
        w = np.random.random(size=500)

        D = distance_matrix_xy(X, Y, w)



