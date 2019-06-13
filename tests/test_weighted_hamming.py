import pytest
import numpy as np

def test_hamming_dist():
    x1 = np.array([1,2,3], dtype=float)
    x2 = np.array([1,0,3], dtype=float)
    w = np.array([1,1,1], dtype=float)

    assert 2 == 3


