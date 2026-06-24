from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import get_centroid, get_pole_of_inaccessibility


class StatsCollector:
    def __init__(self) -> None:
        self.data: list[dict[str, Any]] = []

    def collect(
        self,
        mask: np.ndarray,
        iou_score: float,
        frame_id: int,
        label_id: int,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if mask is None or not np.any(mask):
            return

        centroid = get_centroid(mask)
        pole = get_pole_of_inaccessibility(mask)
        entry = {
            "frame_id": frame_id,
            "mask_label": label_id,
            "area_px": int(np.sum(mask)),
            "iou_score": round(float(iou_score), 4),
            "centroid_x": centroid[0] if centroid else None,
            "centroid_y": centroid[1] if centroid else None,
            "pole_x": pole[0] if pole else None,
            "pole_y": pole[1] if pole else None,
        }
        if meta:
            entry.update(meta)
        self.data.append(entry)

    def get_data(self) -> list[dict[str, Any]]:
        return self.data

    def clear(self) -> None:
        self.data = []
