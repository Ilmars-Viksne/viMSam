from pathlib import Path

import numpy as np

from vimsam_segmenter.utils.standard_stats import build_standard_stats_record
from vimsam_segmenter.utils.stats_schema import STANDARD_STATS_COLUMNS


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
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True

    record = build_standard_stats_record(
        source_path=Path("20260226213602.raw"),
        time_seconds=0.0,
        frame_id=0,
        mask=mask,
        iou_score=0.797123,
        has_combined=True,
    )

    assert record["source_name"] == "20260226213602.raw"
    assert record["time_seconds"] == 0.0
    assert record["frame_id"] == 0
    assert record["mask_label"] == 1
    assert record["area_px"] == 12
    assert record["iou_score"] == 0.7971
    assert record["bbox_x1"] == 3
    assert record["bbox_y1"] == 2
    assert record["bbox_x2"] == 6
    assert record["bbox_y2"] == 4
    assert record["mask_name"] == "frame_00000.png"
    assert record["combined_name"] == "frame_00000_combined.png"
