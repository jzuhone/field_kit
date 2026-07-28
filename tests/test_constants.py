import numpy as np

from field_kit.constants import sqrt2, two_pi


def test_two_pi():
    assert two_pi == 2.0 * np.pi


def test_sqrt2():
    assert sqrt2 == np.sqrt(2.0)
