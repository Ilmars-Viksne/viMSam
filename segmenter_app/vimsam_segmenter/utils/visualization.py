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
) -> np.ndarray:
    image_disp = _normalize_display(image)
    h, w = image_disp.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, raw_mask in enumerate(_extract_masks(segmentation_result)):
        mask = _normalize_mask(raw_mask)
        if mask.shape[:2] != (h, w):
            raise InputValidationError(
                f"Mask shape {mask.shape[:2]} does not match image shape {(h, w)}"
            )
        color = STABLE_PALETTE[(i % 4) + 1]
        colored_mask[mask > 0] = color

    if not save_combined:
        return colored_mask

    overlay = image_disp.copy()
    mask_indices = np.any(colored_mask > 0, axis=-1)
    overlay[mask_indices] = (overlay[mask_indices] * 0.6 + colored_mask[mask_indices] * 0.4).astype(np.uint8)

    fig = Figure(figsize=(18, 6))
    FigureCanvasAgg(fig)
    ax1, ax2, ax3 = fig.subplots(1, 3)

    ax1.imshow(image_disp)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(colored_mask)
    ax2.set_title("Segmentation Mask")
    ax2.axis("off")

    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis("off")

    if prompts:
        prompt_type = prompts.get("type")
        prompt_data = prompts.get("data")
        if prompt_type == "point" and prompt_data is not None:
            pts = np.asarray(prompt_data)
            if pts.ndim == 1:
                pts = pts[None, :]
            ax3.scatter(pts[:, 0], pts[:, 1], marker="x", c="white", s=100, linewidth=2)
        elif prompt_type == "box" and prompt_data is not None:
            b = np.asarray(prompt_data)
            rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=2, edgecolor="white", facecolor="none")
            ax3.add_patch(rect)

    fig.tight_layout()
    fig.canvas.draw()
    im_array = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    fig.clear()
    return np.ascontiguousarray(im_array)


save_masks_as_image = create_visualization
