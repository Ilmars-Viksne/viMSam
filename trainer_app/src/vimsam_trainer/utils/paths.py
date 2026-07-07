from __future__ import annotations

from pathlib import Path

from ..core.errors import OutputWriteError


def ensure_output_dir(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        raise OutputWriteError(f"Output path exists but is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
