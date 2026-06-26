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

    # deterministic overlay blending
    overlay = (image_disp.astype(np.float32) * 0.65 + colored_mask.astype(np.float32) * 0.35)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # draw prompts only when requested
    if show_prompts and prompts:
        prompt_type = prompts.get("type")
        prompt_data = prompts.get("data")

        def _draw_cross(img: np.ndarray, x: int, y: int, color=(255, 255, 255), size: int = 5):
            x = int(round(x))
            y = int(round(y))
            h_, w_ = img.shape[:2]
            for dx in range(-size, size + 1):
                xx = x + dx
                if 0 <= xx < w_ and 0 <= y < h_:
                    img[y, xx] = color
            for dy in range(-size, size + 1):
                yy = y + dy
                if 0 <= yy < h_ and 0 <= x < w_:
                    img[yy, x] = color

        def _draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color=(255, 255, 255), thickness: int = 2):
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            h_, w_ = img.shape[:2]
            x1, x2 = max(0, min(x1, w_ - 1)), max(0, min(x2, w_ - 1))
            y1, y2 = max(0, min(y1, h_ - 1)), max(0, min(y2, h_ - 1))
            for t in range(thickness):
                # top/bottom
                if y1 + t <= y2 - t:
                    img[y1 + t, x1:x2 + 1] = color
                    img[y2 - t, x1:x2 + 1] = color
                # left/right
                if x1 + t <= x2 - t:
                    img[y1:y2 + 1, x1 + t] = color
                    img[y1:y2 + 1, x2 - t] = color

        if prompt_type == "point" and prompt_data is not None:
            pts = np.asarray(prompt_data)
            if pts.ndim == 1:
                pts = pts[None, :]
            for p in pts:
                _draw_cross(overlay, p[0], p[1])
        elif prompt_type == "box" and prompt_data is not None:
            b = np.asarray(prompt_data)
            if b.size >= 4:
                _draw_box(overlay, b[0], b[1], b[2], b[3])

    # side-by-side: original | mask | overlay
    combined = np.concatenate([image_disp, colored_mask, overlay], axis=1)
    return np.ascontiguousarray(combined)


save_masks_as_image = create_visualization
