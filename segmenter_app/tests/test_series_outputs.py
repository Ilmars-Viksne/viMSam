from pathlib import Path

import pytest

from vimsam_segmenter.io.series_outputs import (
    combined_dir,
    combined_frame_path,
    combined_name,
    combined_video_path,
    frame_stem,
    mask_name,
    mask_frame_path,
    masks_dir,
)


def test_frame_stem_uses_five_digits_by_default():
    assert frame_stem(0) == "frame_00000"
    assert frame_stem(1) == "frame_00001"
    assert frame_stem(42) == "frame_00042"
    assert frame_stem(1234) == "frame_01234"


def test_frame_stem_rejects_negative_index():
    with pytest.raises(ValueError):
        frame_stem(-1)


def test_frame_names():
    assert mask_name(0) == "frame_00000.png"
    assert combined_name(0) == "frame_00000_combined.png"


def test_series_directories():
    output_dir = Path("output")

    assert masks_dir(output_dir) == Path("output/masks")
    assert combined_dir(output_dir) == Path("output/combined")


def test_frame_paths():
    output_dir = Path("output")

    assert mask_frame_path(output_dir, 0) == Path("output/masks/frame_00000.png")
    assert combined_frame_path(output_dir, 0) == Path("output/combined/frame_00000_combined.png")


def test_combined_video_path():
    assert combined_video_path(Path("output")) == Path("output/combined_video.mp4")
