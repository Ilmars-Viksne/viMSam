from __future__ import annotations

from pathlib import Path


DEFAULT_FRAME_DIGITS = 5


def frame_stem(frame_index: int, digits: int = DEFAULT_FRAME_DIGITS) -> str:
    """Return the canonical frame stem for all series workflows."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    return f"frame_{frame_index:0{digits}d}"


def masks_dir(output_dir: Path) -> Path:
    """Directory for clean mask frames."""
    return output_dir / "masks"


def combined_dir(output_dir: Path) -> Path:
    """Directory for combined diagnostic frames."""
    return output_dir / "combined"


def ensure_series_output_dirs(
    output_dir: Path,
    *,
    save_combined: bool = False,
) -> tuple[Path, Path | None]:
    """Create the standard output directories for series workflows."""
    mask_output_dir = masks_dir(output_dir)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    combined_output_dir: Path | None = None
    if save_combined:
        combined_output_dir = combined_dir(output_dir)
        combined_output_dir.mkdir(parents=True, exist_ok=True)

    return mask_output_dir, combined_output_dir


def mask_frame_path(output_dir: Path, frame_index: int) -> Path:
    """Path for a clean mask frame."""
    return masks_dir(output_dir) / f"{frame_stem(frame_index)}.png"


def combined_frame_path(output_dir: Path, frame_index: int) -> Path:
    """Path for a combined diagnostic frame."""
    return combined_dir(output_dir) / f"{frame_stem(frame_index)}_combined.png"


def combined_video_path(output_dir: Path) -> Path:
    """Path for the optional combined diagnostic MP4."""
    return output_dir / "combined_video.mp4"
