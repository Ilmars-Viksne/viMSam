from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io.series_outputs import combined_name, mask_name
from .geometry import get_box_from_mask, get_centroid, get_pole_of_inaccessibility


def _safe_float(value: Any) -> float | str:
    if value is None:
        return ""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return ""


def build_standard_stats_record(
    *,
    source_path: Path | None,
    time_seconds: float | None,
    frame_id: int,
    mask: np.ndarray,
    mask_label: int = 1,
    iou_score: float | None = None,
    has_combined: bool = False,
) -> dict[str, Any]:
    mask_arr = np.asarray(mask)
    binary_mask = mask_arr > 0

    centroid = get_centroid(binary_mask)
    pole = get_pole_of_inaccessibility(binary_mask)
    bbox = get_box_from_mask(binary_mask, padding=0)

    if bbox is not None:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(value) for value in bbox]
    else:
        bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = ""

    return {
        "source_name": source_path.name if source_path is not None else "",
        "time_seconds": _safe_float(time_seconds),
        "frame_id": int(frame_id),
        "mask_label": int(mask_label),
        "area_px": int(np.count_nonzero(binary_mask)),
        "iou_score": _safe_float(iou_score),
        "centroid_x": centroid[0] if centroid is not None else "",
        "centroid_y": centroid[1] if centroid is not None else "",
        "pole_x": pole[0] if pole is not None else "",
        "pole_y": pole[1] if pole is not None else "",
        "bbox_x1": bbox_x1,
        "bbox_y1": bbox_y1,
        "bbox_x2": bbox_x2,
        "bbox_y2": bbox_y2,
        "mask_name": mask_name(frame_id),
        "combined_name": combined_name(frame_id) if has_combined else "",
    }
