from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label


def _as_mask(mask: object) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size == 0 or arr.ndim < 2:
        return None
    return arr


def get_centroid(mask: object) -> tuple[int, int] | None:
    arr = _as_mask(mask)
    if arr is None:
        return None
    y_indices, x_indices = np.where(arr > 0)
    if len(y_indices) == 0:
        return None
    return (int(np.mean(x_indices)), int(np.mean(y_indices)))


def get_pole_of_inaccessibility(mask: object) -> tuple[int, int] | None:
    arr = _as_mask(mask)
    if arr is None or not np.any(arr):
        return None
    labeled_mask, num_features = label(arr > 0)
    if num_features == 0:
        return None
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    largest_component = labeled_mask == component_sizes.argmax()
    dist_map = distance_transform_edt(largest_component)
    cy, cx = np.unravel_index(int(np.argmax(dist_map)), dist_map.shape)
    return (int(cx), int(cy))


def get_box_from_mask(mask: object, padding: int = 20) -> np.ndarray | None:
    arr = _as_mask(mask)
    if arr is None:
        return None
    y_indices, x_indices = np.where(arr > 0)
    if len(y_indices) == 0:
        return None
    height, width = arr.shape[:2]
    x1 = max(0, int(x_indices.min()) - padding)
    y1 = max(0, int(y_indices.min()) - padding)
    x2 = min(width, int(x_indices.max()) + padding)
    y2 = min(height, int(y_indices.max()) + padding)
    return np.array([x1, y1, x2, y2])
