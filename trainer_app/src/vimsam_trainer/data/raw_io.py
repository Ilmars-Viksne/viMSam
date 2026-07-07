from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import InputValidationError


def _validate_dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int):
        raise InputValidationError("Raw width and height must be integers")
    if width <= 0 or height <= 0:
        raise InputValidationError("Raw width and height must be positive")


def validate_raw_file_size(path: Path, width: int, height: int) -> int:
    _validate_dimensions(width, height)
    if not path.is_file():
        raise InputValidationError(f"Raw file does not exist: {path}")

    expected_size = width * height * np.dtype(np.uint16).itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise InputValidationError(
            f"Raw file size mismatch for {path}: "
            f"expected {expected_size} bytes "
            f"({width}x{height} uint16), got {actual_size} bytes"
        )
    return actual_size


def read_u16_raw(path: Path, width: int, height: int) -> np.ndarray:
    validate_raw_file_size(path, width, height)
    try:
        data = np.fromfile(path, dtype=np.uint16)
    except OSError as exc:
        raise InputValidationError(f"Could not read raw file {path}: {exc}") from exc
    try:
        return data.reshape((height, width))
    except ValueError as exc:
        raise InputValidationError(
            f"Could not reshape raw file {path} to {height}x{width}"
        ) from exc
