from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.errors import InputValidationError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
RAW_EXTENSIONS = {".raw"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True, slots=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path


def _ensure_dir(path: Path, label: str) -> Path:
    if path.exists() and not path.is_dir():
        raise InputValidationError(f"{label} path exists but is not a directory: {path}")
    if not path.is_dir():
        raise InputValidationError(f"{label} directory does not exist: {path}")
    return path


def _allowed_image_extensions(workflow: str) -> set[str]:
    if workflow == "raw_frames":
        return RAW_EXTENSIONS
    if workflow == "image_frames":
        return IMAGE_EXTENSIONS
    raise InputValidationError(f"Unsupported workflow: {workflow}")


def _collect_files(path: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in extensions
    )


def _collect_masks_by_stem(path: Path) -> dict[str, Path]:
    masks_by_stem: dict[str, Path] = {}
    for mask in _collect_files(path, MASK_EXTENSIONS):
        if mask.stem in masks_by_stem:
            raise InputValidationError(f"Duplicate mask stem {mask.stem!r} in {path}")
        masks_by_stem[mask.stem] = mask
    return masks_by_stem


def collect_image_mask_pairs(
    *,
    images_path: Path,
    masks_path: Path,
    workflow: str,
) -> list[ImageMaskPair]:
    images_path = _ensure_dir(images_path, "Images")
    masks_path = _ensure_dir(masks_path, "Masks")

    images = _collect_files(images_path, _allowed_image_extensions(workflow))
    masks_by_stem = _collect_masks_by_stem(masks_path)

    return [
        ImageMaskPair(image_path=image, mask_path=masks_by_stem[image.stem])
        for image in images
        if image.stem in masks_by_stem
    ]
