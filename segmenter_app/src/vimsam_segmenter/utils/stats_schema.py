from __future__ import annotations

from typing import Any


STANDARD_STATS_COLUMNS = [
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


def ordered_stats_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **{column: record.get(column, "") for column in STANDARD_STATS_COLUMNS},
        **{column: value for column, value in record.items() if column not in STANDARD_STATS_COLUMNS},
    }


def ordered_stats_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extra_columns: list[str] = []
    for record in records:
        for column in record:
            if column not in STANDARD_STATS_COLUMNS and column not in extra_columns:
                extra_columns.append(column)

    ordered_columns = STANDARD_STATS_COLUMNS + extra_columns
    return [
        {column: record.get(column, "") for column in ordered_columns}
        for record in records
    ]
