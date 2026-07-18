from pathlib import Path

import numpy as np

from vimsam_segmenter.utils.standard_stats import (
    build_standard_stats_record,
)
from vimsam_segmenter.utils.stats_schema import (
    STANDARD_STATS_COLUMNS,
)


def make_test_mask() -> np.ndarray:
    """Create a 4 x 3 rectangular mask with an area of 12 pixels."""

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    return mask


def test_standard_stats_column_order():
    assert STANDARD_STATS_COLUMNS == [
        "source_name",
        "time_seconds",
        "frame_id",
        "mask_label",
        "area_px",
        "iou_score",
        "centroid_x",
        "centroid_y",
        "pole_x",
        "pole_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mask_name",
        "combined_name",
    ]


def test_build_standard_stats_record():
    """Verify legacy fallback names when explicit names are omitted."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=0,
        mask=mask,
        mask_label=1,
        iou_score=0.7971,
        has_combined=True,
    )

    assert record["source_name"] == "20260226213602.raw"
    assert record["time_seconds"] == 0.0
    assert record["frame_id"] == 0
    assert record["mask_label"] == 1
    assert record["area_px"] == 12
    assert record["iou_score"] == 0.7971

    assert record["centroid_x"] == 4
    assert record["centroid_y"] == 3
    assert record["pole_x"] != ""
    assert record["pole_y"] != ""

    assert record["bbox_x1"] == 3
    assert record["bbox_y1"] == 2
    assert record["bbox_x2"] == 6
    assert record["bbox_y2"] == 4

    assert record["mask_name"] == "frame_00000.png"
    assert (
        record["combined_name"]
        == "frame_00000_combined.png"
    )


def test_build_standard_stats_record_without_combined():
    """The combined name must be empty when combined output is disabled."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=0,
        mask=mask,
        mask_label=1,
        iou_score=0.7971,
        has_combined=False,
    )

    assert record["mask_name"] == "frame_00000.png"
    assert record["combined_name"] == ""


def test_build_standard_stats_record_uses_explicit_source_names():
    """Explicit source-based output names must override legacy names."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=0,
        mask=mask,
        mask_label=1,
        iou_score=0.7971,
        has_combined=True,
        mask_output_name="20260226213602.png",
        combined_output_name=(
            "20260226213602_combined.png"
        ),
    )

    assert record["mask_name"] == "20260226213602.png"
    assert (
        record["combined_name"]
        == "20260226213602_combined.png"
    )


def test_build_standard_stats_record_uses_prefixed_source_names():
    """Explicit prefix-source names must be preserved unchanged."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=0,
        mask=mask,
        mask_label=1,
        iou_score=0.7971,
        has_combined=True,
        mask_output_name=(
            "experiment_a_20260226213602.png"
        ),
        combined_output_name=(
            "experiment_a_20260226213602_combined.png"
        ),
    )

    assert (
        record["mask_name"]
        == "experiment_a_20260226213602.png"
    )
    assert (
        record["combined_name"]
        == "experiment_a_20260226213602_combined.png"
    )


def test_build_standard_stats_record_uses_eight_digit_index_names():
    """Explicit eight-digit index names must be preserved unchanged."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=0,
        mask=mask,
        mask_label=1,
        iou_score=0.7971,
        has_combined=True,
        mask_output_name="00000000.png",
        combined_output_name="00000000_combined.png",
    )

    assert record["mask_name"] == "00000000.png"
    assert record["combined_name"] == "00000000_combined.png"


def test_build_standard_stats_record_uses_prefixed_index_names():
    """Explicit prefix-index names must be preserved unchanged."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=42,
        mask=mask,
        mask_label=1,
        iou_score=0.7971,
        has_combined=True,
        mask_output_name="experiment_a_00000042.png",
        combined_output_name=(
            "experiment_a_00000042_combined.png"
        ),
    )

    assert record["frame_id"] == 42
    assert (
        record["mask_name"]
        == "experiment_a_00000042.png"
    )
    assert (
        record["combined_name"]
        == "experiment_a_00000042_combined.png"
    )


def test_explicit_combined_name_is_ignored_when_not_saved():
    """A combined filename must not be reported when it was not saved."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=0,
        mask=mask,
        has_combined=False,
        mask_output_name="20260226213602.png",
        combined_output_name=(
            "20260226213602_combined.png"
        ),
    )

    assert record["mask_name"] == "20260226213602.png"
    assert record["combined_name"] == ""


def test_build_standard_stats_record_handles_empty_mask():
    """Geometry fields must be empty for a mask containing no objects."""

    mask = np.zeros((10, 10), dtype=bool)

    record = build_standard_stats_record(
        source_path=Path("empty.raw"),
        time_seconds=1.5,
        frame_id=1,
        mask=mask,
        has_combined=False,
        mask_output_name="empty.png",
    )

    assert record["source_name"] == "empty.raw"
    assert record["time_seconds"] == 1.5
    assert record["frame_id"] == 1
    assert record["area_px"] == 0

    assert record["centroid_x"] == ""
    assert record["centroid_y"] == ""
    assert record["pole_x"] == ""
    assert record["pole_y"] == ""

    assert record["bbox_x1"] == ""
    assert record["bbox_y1"] == ""
    assert record["bbox_x2"] == ""
    assert record["bbox_y2"] == ""

    assert record["mask_name"] == "empty.png"
    assert record["combined_name"] == ""


def test_build_standard_stats_record_handles_missing_source():
    """A missing source path must produce an empty source name."""

    mask = make_test_mask()

    record = build_standard_stats_record(
        source_path=None,
        time_seconds=None,
        frame_id=0,
        mask=mask,
        iou_score=None,
        has_combined=False,
    )

    assert record["source_name"] == ""
    assert record["time_seconds"] == ""
    assert record["iou_score"] == ""
    assert record["mask_name"] == "frame_00000.png"
    assert record["combined_name"] == ""