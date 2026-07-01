from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import numpy as np

from ..core.errors import InputValidationError

STABLE_PALETTE = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
    dtype=np.uint8,
)


def _normalize_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.size == 0:
        raise InputValidationError("Cannot visualize an empty image")
    image = np.squeeze(image)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    elif image.ndim != 3 or image.shape[-1] not in {1, 3}:
        raise InputValidationError(f"Expected a 2D grayscale or RGB image, got shape {image.shape}")
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    return ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)


def _extract_masks(segmentation_result: object) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if isinstance(segmentation_result, list):
        for mask_entry in segmentation_result:
            if isinstance(mask_entry, dict) and "segmentation" in mask_entry:
                masks.append(np.asarray(mask_entry["segmentation"]))
    elif isinstance(segmentation_result, np.ndarray):
        if segmentation_result.ndim == 3:
            masks.extend(np.asarray(segmentation_result[i]) for i in range(segmentation_result.shape[0]))
        else:
            masks.append(np.asarray(segmentation_result))
    return masks


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise InputValidationError(f"Expected a 2D mask, got shape {mask.shape}")
    return mask


def _draw_prompts_on_image(image: np.ndarray, prompts: dict[str, np.ndarray] | None) -> np.ndarray:
    if not prompts:
        return image

    image_disp = np.array(image, copy=True)
    points = prompts.get("points")
    if points is not None:
        points_arr = np.asarray(points)
        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(1, 2)
        for x, y in points_arr:
            x = int(round(float(x)))
            y = int(round(float(y)))
            for dx in range(-4, 5):
                xx = x + dx
                if 0 <= xx < image_disp.shape[1] and 0 <= y < image_disp.shape[0]:
                    image_disp[y, xx] = (255, 255, 255)
            for dy in range(-4, 5):
                yy = y + dy
                if 0 <= yy < image_disp.shape[0] and 0 <= x < image_disp.shape[1]:
                    image_disp[yy, x] = (255, 255, 255)

    box = prompts.get("box")
    if box is not None:
        box_arr = np.asarray(box).reshape(4)
        x1, y1, x2, y2 = [int(round(float(v))) for v in box_arr]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(x1, image_disp.shape[1] - 1))
        x2 = max(0, min(x2, image_disp.shape[1] - 1))
        y1 = max(0, min(y1, image_disp.shape[0] - 1))
        y2 = max(0, min(y2, image_disp.shape[0] - 1))
        image_disp[y1:y2 + 1, x1] = (255, 255, 255)
        image_disp[y1:y2 + 1, x2] = (255, 255, 255)
        image_disp[y1, x1:x2 + 1] = (255, 255, 255)
        image_disp[y2, x1:x2 + 1] = (255, 255, 255)

    return image_disp


def create_visualization(
    image: np.ndarray,
    segmentation_result: object,
    prompts: dict[str, np.ndarray] | None = None,
    save_combined: bool = False,
    show_prompts: bool = False,
) -> np.ndarray:
    image_disp = _normalize_display(image)
    h, w = image_disp.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    masks = _extract_masks(segmentation_result)
    for idx, raw_mask in enumerate(masks, start=1):
        mask_2d = _normalize_mask(raw_mask)
        if mask_2d.shape != (h, w):
            raise InputValidationError(
                f"Mask shape {mask_2d.shape} does not match image shape {(h, w)}"
            )

        color = STABLE_PALETTE[idx % len(STABLE_PALETTE)]
        colored_mask[mask_2d > 0] = color

    if not save_combined:
        return np.ascontiguousarray(colored_mask)

    overlay = (image_disp.astype(np.float32) * 0.65 + colored_mask.astype(np.float32) * 0.35)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if show_prompts:
        overlay = _draw_prompts_on_image(overlay, prompts)

    # side-by-side: original | mask | overlay
    combined = np.concatenate([image_disp, colored_mask, overlay], axis=1)
    return np.ascontiguousarray(combined)


save_masks_as_image = create_visualization
