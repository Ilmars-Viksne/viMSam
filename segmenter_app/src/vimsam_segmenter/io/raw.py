from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import InputValidationError
from .local import ensure_input_dir, ensure_input_file


def _validate_dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int):
        raise InputValidationError("Raw width and height must be integers")
    if width <= 0 or height <= 0:
        raise InputValidationError("Raw width and height must be positive")


def validate_raw_file_size(path: Path, width: int = 1024, height: int = 1024) -> int:
    _validate_dimensions(width, height)
    path = ensure_input_file(path)
    expected_size = width * height * 2
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise InputValidationError(
            f"Raw file size mismatch for {path}: expected {expected_size} bytes "
            f"({width}x{height}x16-bit), got {actual_size} bytes"
        )
    return actual_size


def read_u3cmos_raw(path: Path, width: int = 1024, height: int = 1024) -> np.ndarray:
    path = ensure_input_file(path)
    validate_raw_file_size(path, width=width, height=height)
    try:
        raw_data = path.read_bytes()
    except OSError as exc:
        raise InputValidationError(f"Could not read raw file {path}: {exc}") from exc

    image_1d = np.frombuffer(raw_data, dtype=np.uint16)
    if image_1d.size != width * height:
        raise InputValidationError(
            f"Raw pixel count mismatch for {path}: expected {width * height}, got {image_1d.size}"
        )
    image_2d = image_1d.reshape((height, width)).astype(np.float32)
    image_2d = np.flipud(image_2d)

    p1, p99 = np.percentile(image_2d, (1, 99))
    image_2d = np.clip(image_2d, p1, p99)
    if p99 > p1:
        image_normalized = (image_2d - p1) / (p99 - p1) * 255.0
    else:
        image_normalized = np.zeros_like(image_2d)
    return image_normalized.astype(np.uint8)


def get_raw_timeseries_files(directory: Path) -> list[Path]:
    directory = ensure_input_dir(directory)
    files = sorted(path for path in directory.iterdir() if path.is_file())
    raw_files = [path for path in files if path.suffix.lower() == ".raw"]
    return raw_files if raw_files else files


def validate_raw_timeseries_files(files: list[Path], width: int = 1024, height: int = 1024) -> None:
    if not files:
        raise InputValidationError("No raw files found")
    for path in files:
        validate_raw_file_size(path, width=width, height=height)
