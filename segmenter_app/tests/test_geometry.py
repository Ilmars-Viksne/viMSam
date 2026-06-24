import numpy as np

from src.utils.geometry import get_box_from_mask, get_centroid, get_pole_of_inaccessibility


def test_geometry_empty_mask_returns_none():
    mask = np.zeros((5, 5), dtype=bool)

    assert get_centroid(mask) is None
    assert get_pole_of_inaccessibility(mask) is None
    assert get_box_from_mask(mask) is None


def test_geometry_non_empty_mask():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True

    assert get_centroid(mask) == (4, 3)
    assert get_pole_of_inaccessibility(mask) is not None
    assert get_box_from_mask(mask, padding=1).tolist() == [2, 1, 7, 5]
