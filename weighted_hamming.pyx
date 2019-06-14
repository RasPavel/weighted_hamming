import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
def distance_matrix(DTYPE_t [:,:] X, DTYPE_t [:] w):
    cdef int i,j
    cdef int n_points = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] D = np.zeros((n_points, n_points), DTYPE)

    with nogil:
        for i in prange(n_points, schedule='static', chunksize=1):
            for j in range(i, n_points):
                D[j,i] = D[i,j] = hamming_dist(X[i], X[j], w, n_features)
    return D


@cython.boundscheck(False)
@cython.wraparound(False)
def distance_matrix_xy(DTYPE_t [:,:] X, DTYPE_t [:,:] Y, DTYPE_t [:] w):
    cdef int i,j
    cdef int N = X.shape[0]
    cdef int M = Y.shape[0]
    cdef int n_features = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] D = np.zeros((N, M), DTYPE)

    with nogil:
        for i in prange(N, schedule='static'):
            for j in range(M):
                D[i,j] = hamming_dist(X[i], Y[j], w, n_features)

    return D


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t hamming_dist(DTYPE_t [:] x1, DTYPE_t [:] x2,
                   DTYPE_t [:] w, int size) nogil:
    cdef DTYPE_t d = 0
    cdef int j
    for j in range(size):
        if x1[j] != x2[j]:
            d += w[j]
    return d / size