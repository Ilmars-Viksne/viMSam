from typing import List, Dict, Any
from .geometry import get_centroid, get_pole_of_inaccessibility
import numpy as np

class StatsCollector:
    def __init__(self):
        self.data: List[Dict[str, Any]] = []

    def collect(self, mask, iou_score, frame_id, label_id, meta=None):
        if mask is None or not np.any(mask):
            return

        centroid = get_centroid(mask)
        pole = get_pole_of_inaccessibility(mask)
        area = int(np.sum(mask))

        entry = {
            "frame_id": frame_id,
            "mask_label": label_id,
            "area_px": area,
            "iou_score": round(iou_score, 4),
            "centroid_x": centroid[0] if centroid else None,
            "centroid_y": centroid[1] if centroid else None,
            "pole_x": pole[0] if pole else None,
            "pole_y": pole[1] if pole else None
        }
        if meta:
            entry.update(meta)

        self.data.append(entry)

    def get_data(self) -> List[Dict]:
        return self.data

    def clear(self):
        self.data = []
