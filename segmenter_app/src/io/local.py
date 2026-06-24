from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Generator

from ..core.config import normalize_path
from ..core.errors import DependencyMissingError, InputValidationError, OutputWriteError


def _imageio():
    try:
        import imageio.v3 as imageio
    except ImportError as exc:
        raise DependencyMissingError("imageio is required for image and video I/O.") from exc
    return imageio


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyMissingError("numpy is required for image and video I/O.") from exc
    return np


def _describe(path: Path) -> str:
    return str(path)


def _ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise OutputWriteError(f"Output parent exists but is not a directory: {_describe(parent)}")
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(f"Could not create output directory {_describe(parent)}: {exc}") from exc


def ensure_input_file(path: Path) -> Path:
    path = normalize_path(path)
    if path.exists() and not path.is_file():
        raise InputValidationError(f"Expected an input file but found a directory: {_describe(path)}")
    if not path.is_file():
        raise InputValidationError(f"Input file does not exist: {_describe(path)}")
    return path


def ensure_input_dir(path: Path) -> Path:
    path = normalize_path(path)
    if path.exists() and not path.is_dir():
        raise InputValidationError(f"Expected an input directory but found a file: {_describe(path)}")
    if not path.is_dir():
        raise InputValidationError(f"Input directory does not exist: {_describe(path)}")
    return path


def output_dir_for(path: Path) -> Path:
    path = normalize_path(path)
    output_dir = path if path.exists() and path.is_dir() else path.parent if path.suffix else path
    if output_dir.exists() and not output_dir.is_dir():
        raise OutputWriteError(f"Output directory path exists but is not a directory: {_describe(output_dir)}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(f"Could not create output directory {_describe(output_dir)}: {exc}") from exc
    return output_dir


def default_image_output_name(input_path: Path) -> str:
    return f"res_{input_path.stem}.png"


def resolve_image_output(output_path: Path, input_path: Path) -> Path:
    output_path = normalize_path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / default_image_output_name(input_path)
    if output_path.suffix:
        _ensure_parent_dir(output_path)
        return output_path
    if output_path.exists() and not output_path.is_dir():
        raise OutputWriteError(f"Output path exists but is not a directory: {_describe(output_path)}")
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(f"Could not create output directory {_describe(output_path)}: {exc}") from exc
    return output_path / default_image_output_name(input_path)


def sibling_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def load_image(path: Path) -> np.ndarray:
    path = ensure_input_file(path)
    try:
        return _imageio().imread(path)
    except DependencyMissingError:
        raise
    except Exception as exc:
        raise InputValidationError(f"Could not read image {_describe(path)}: {exc}") from exc


def stream_video(path: Path) -> Generator[np.ndarray, None, None]:
    path = ensure_input_file(path)
    try:
        yield from _imageio().imiter(path)
    except DependencyMissingError:
        raise
    except Exception as exc:
        raise InputValidationError(f"Could not read video {_describe(path)}: {exc}") from exc


def read_metadata(path: Path) -> dict[str, Any]:
    path = ensure_input_file(path)
    try:
        return dict(_imageio().immeta(path))
    except DependencyMissingError:
        raise
    except Exception:
        return {}


def list_files(path: Path, patterns: tuple[str, ...]) -> list[Path]:
    path = normalize_path(path)
    if path.is_file():
        return [path]
    ensure_input_dir(path)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(p for p in files if p.is_file())


def save_image(path: Path, image: np.ndarray) -> Path:
    path = normalize_path(path)
    _ensure_parent_dir(path)
    try:
        imageio = _imageio()
        np = _numpy()
        imageio.imwrite(path, np.ascontiguousarray(image))
    except DependencyMissingError:
        raise
    except Exception as exc:
        raise OutputWriteError(f"Could not write image {_describe(path)}: {exc}") from exc
    return path


def save_video(path: Path, frames: list[np.ndarray], fps: int = 5) -> Path:
    if not frames:
        raise OutputWriteError(f"Cannot write video with no frames: {_describe(Path(path))}")
    path = normalize_path(path)
    _ensure_parent_dir(path)
    try:
        _imageio().imwrite(path, frames, fps=fps, codec="libx264")
    except DependencyMissingError:
        raise
    except Exception as exc:
        raise OutputWriteError(f"Could not write video {_describe(path)}: {exc}") from exc
    return path


def save_records(path: Path, records: list[dict[str, Any]], export_format: str) -> Path | None:
    if not records:
        return None
    if export_format not in {"csv", "json"}:
        raise InputValidationError("format must be one of: csv, json")
    path = normalize_path(path).with_suffix(f".{export_format}")
    _ensure_parent_dir(path)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=f".tmp.{export_format}") as handle:
            temp_path = Path(handle.name)
            if export_format == "csv":
                try:
                    import pandas as pd
                except ImportError as exc:
                    raise DependencyMissingError("pandas is required to write CSV stats.") from exc
                pd.DataFrame(records).to_csv(handle, index=False)
            else:
                json.dump(records, handle, indent=4)
                handle.write("\n")
        temp_path.replace(path)
    except DependencyMissingError:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OutputWriteError(f"Could not write stats {_describe(path)}: {exc}") from exc
    return path
