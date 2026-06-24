import numpy as np
import pytest

from src.core.errors import InputValidationError
from src.utils.visualization import create_visualization


def test_mask_visualization_shape_and_dtype():
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True

    result = create_visualization(image, mask)

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8


def test_combined_visualization_is_rgb_uint8():
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True

    result = create_visualization(image, mask, prompts={"type": "point", "data": np.array([[2, 2]])}, save_combined=True)

    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.dtype == np.uint8


def test_visualization_rejects_mismatched_mask_shape():
    image = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)

    with pytest.raises(InputValidationError, match="does not match image shape"):
        create_visualization(image, mask)
