import numpy as np
from scipy.ndimage import distance_transform_edt, label

def get_centroid(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0: return None
    cy = int(np.mean(y_indices))
    cx = int(np.mean(x_indices))
    return (cx, cy)

def get_pole_of_inaccessibility(mask):
    if not np.any(mask): return None
    labeled_mask, num_features = label(mask)
    if num_features == 0: return None

    # Largest component only
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()
    largest_component = (labeled_mask == largest_label)

    dist_map = distance_transform_edt(largest_component)
    flat_idx = np.argmax(dist_map)
    cy, cx = np.unravel_index(flat_idx, dist_map.shape)
    return (int(cx), int(cy))

def get_box_from_mask(mask, padding=20):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0: return None
    H, W = mask.shape[:2]
    x1 = max(0, x_indices.min() - padding)
    y1 = max(0, y_indices.min() - padding)
    x2 = min(W, x_indices.max() + padding)
    y2 = min(H, y_indices.max() + padding)
    return np.array([x1, y1, x2, y2])
