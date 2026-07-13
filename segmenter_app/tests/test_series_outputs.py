from pathlib import Path

import pytest

from vimsam_segmenter.io.series_outputs import (
    combined_frame_path,
    combined_name,
    frame_stem,
    index_stem,
    mask_frame_path,
    mask_name,
    video_source_stem,
)


def test_index_stem_uses_eight_digits():
    assert index_stem(0) == "00000000"
    assert index_stem(1) == "00000001"
    assert index_stem(42) == "00000042"
    assert index_stem(1234) == "00001234"


def test_index_stem_rejects_negative_index():
    with pytest.raises(ValueError):
        index_stem(-1)


def test_source_frame_stem():
    assert frame_stem(
        frame_index=0,
        mode="source",
        source_path=Path("20260226213602.raw"),
    ) == "20260226213602"


def test_prefix_source_frame_stem():
    assert frame_stem(
        frame_index=0,
        mode="prefix-source",
        source_path=Path("20260226213602.raw"),
        prefix="cell_a",
    ) == "cell_a_20260226213602"


def test_index_frame_stem():
    assert frame_stem(
        frame_index=0,
        mode="index",
    ) == "00000000"


def test_prefix_index_frame_stem():
    assert frame_stem(
        frame_index=42,
        mode="prefix-index",
        prefix="cell_a",
    ) == "cell_a_00000042"


def test_mask_names():
    assert mask_name(
        frame_index=0,
        mode="source",
        source_path=Path("sample.tif"),
    ) == "sample.png"

    assert mask_name(
        frame_index=0,
        mode="prefix-source",
        source_path=Path("sample.tif"),
        prefix="experiment",
    ) == "experiment_sample.png"

    assert mask_name(
        frame_index=0,
        mode="index",
    ) == "00000000.png"

    assert mask_name(
        frame_index=0,
        mode="prefix-index",
        prefix="experiment",
    ) == "experiment_00000000.png"


def test_combined_names():
    assert combined_name(
        frame_index=0,
        mode="source",
        source_path=Path("sample.tif"),
    ) == "sample_combined.png"

    assert combined_name(
        frame_index=0,
        mode="index",
    ) == "00000000_combined.png"


def test_series_frame_paths():
    output_dir = Path("output")

    assert mask_frame_path(
        output_dir,
        frame_index=0,
        mode="source",
        source_path=Path("sample.tif"),
    ) == Path("output/masks/sample.png")

    assert combined_frame_path(
        output_dir,
        frame_index=0,
        mode="prefix-index",
        prefix="cell",
    ) == Path(
        "output/combined/cell_00000000_combined.png"
    )


def test_prefix_mode_requires_prefix():
    with pytest.raises(ValueError):
        frame_stem(
            frame_index=0,
            mode="prefix-index",
        )


def test_video_virtual_source_stem():
    assert video_source_stem(
        Path("moving_cell.mp4"),
        0,
    ) == "moving_cell_00000000"

    assert video_source_stem(
        Path("moving_cell.mp4"),
        42,
    ) == "moving_cell_00000042"