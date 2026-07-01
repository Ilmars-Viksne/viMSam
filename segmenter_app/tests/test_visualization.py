import numpy as np
import pytest

from vimsam_segmenter.core.errors import InputValidationError
from vimsam_segmenter.utils.prompts import build_prompt_overlay
from vimsam_segmenter.utils.visualization import create_visualization


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


def test_mask_visualization_never_draws_prompts():
    image = np.zeros((64, 64), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True

    prompts = build_prompt_overlay(points=((30, 30),))

    clean_mask = create_visualization(
        image,
        mask,
        prompts=None,
        save_combined=False,
        show_prompts=False,
    )

    prompted_mask_attempt = create_visualization(
        image,
        mask,
        prompts=prompts,
        save_combined=False,
        show_prompts=True,
    )

    assert np.array_equal(clean_mask, prompted_mask_attempt)


def test_combined_visualization_draws_prompts_when_requested():
    image = np.zeros((64, 64), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True

    prompts = build_prompt_overlay(points=((30, 30),))

    without_prompts = create_visualization(
        image,
        mask,
        prompts=prompts,
        save_combined=True,
        show_prompts=False,
    )

    with_prompts = create_visualization(
        image,
        mask,
        prompts=prompts,
        save_combined=True,
        show_prompts=True,
    )

    assert without_prompts.shape == with_prompts.shape
    assert not np.array_equal(without_prompts, with_prompts)


def test_combined_visualization_accepts_missing_prompts():
    image = np.zeros((64, 64), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True

    combined = create_visualization(
        image,
        mask,
        prompts=None,
        save_combined=True,
        show_prompts=True,
    )

    assert combined.ndim == 3
    assert combined.shape[-1] == 3
    assert combined.dtype == np.uint8
