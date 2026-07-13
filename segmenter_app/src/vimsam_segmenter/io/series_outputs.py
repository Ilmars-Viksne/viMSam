from __future__ import annotations

import re
from pathlib import Path


DEFAULT_FRAME_DIGITS = 8

FRAME_NAME_MODES = {
    "source",
    "prefix-source",
    "index",
    "prefix-index",
}

def validate_unique_frame_names(
    names: list[str],
) -> None:
    """Reject duplicate output filenames before processing starts."""

    seen: set[str] = set()
    duplicates: set[str] = set()

    for name in names:
        normalized = name.casefold()

        if normalized in seen:
            duplicates.add(name)
        else:
            seen.add(normalized)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Duplicate output frame names would be generated: "
            f"{duplicate_list}"
        )

def sanitize_filename_stem(value: str) -> str:
    """Return a filesystem-safe filename stem.

    The function preserves letters, numbers, underscores, hyphens,
    dots, and spaces. Other characters are replaced with underscores.
    """

    value = value.strip()

    # Prevent path components from appearing in a filename.
    value = Path(value).name

    # Replace characters that are unsafe on common filesystems.
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)

    # Avoid trailing spaces/dots, problematic on Windows.
    value = value.rstrip(" .")

    if not value:
        raise ValueError("The resulting filename stem is empty.")

    return value


def index_stem(
    frame_index: int,
    digits: int = DEFAULT_FRAME_DIGITS,
) -> str:
    """Return a zero-based frame index padded to the requested width."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    if digits <= 0:
        raise ValueError("digits must be positive")

    return f"{frame_index:0{digits}d}"


def video_source_stem(
    video_path: Path,
    frame_index: int,
) -> str:
    """Return a virtual source-frame stem for a decoded video frame.

    Individual decoded video frames do not have source filenames.
    Therefore, the video stem and an eight-digit index are combined.
    """

    return (
        f"{sanitize_filename_stem(video_path.stem)}_"
        f"{index_stem(frame_index)}"
    )


def frame_stem(
    *,
    frame_index: int,
    mode: str = "source",
    source_path: Path | None = None,
    source_stem: str | None = None,
    prefix: str | None = None,
) -> str:
    """Build the output stem for a series frame.

    Modes:
        source:
            SOURCE

        prefix-source:
            PREFIX_SOURCE

        index:
            00000000

        prefix-index:
            PREFIX_00000000
    """

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    if mode not in FRAME_NAME_MODES:
        allowed = ", ".join(sorted(FRAME_NAME_MODES))
        raise ValueError(f"mode must be one of: {allowed}")

    clean_prefix = None

    if prefix is not None:
        clean_prefix = sanitize_filename_stem(prefix)

    if mode in {"prefix-source", "prefix-index"}:
        if clean_prefix is None:
            raise ValueError(
                f"prefix is required for frame naming mode '{mode}'"
            )

    if source_stem is None and source_path is not None:
        source_stem = source_path.stem

    if mode in {"source", "prefix-source"}:
        if source_stem is None:
            raise ValueError(
                f"source_stem or source_path is required "
                f"for frame naming mode '{mode}'"
            )

        base_stem = sanitize_filename_stem(source_stem)

    else:
        base_stem = index_stem(frame_index)

    if mode in {"prefix-source", "prefix-index"}:
        return f"{clean_prefix}_{base_stem}"

    return base_stem


def mask_name(
    *,
    frame_index: int,
    mode: str = "source",
    source_path: Path | None = None,
    source_stem: str | None = None,
    prefix: str | None = None,
) -> str:
    """Return the filename for a clean mask frame."""

    stem = frame_stem(
        frame_index=frame_index,
        mode=mode,
        source_path=source_path,
        source_stem=source_stem,
        prefix=prefix,
    )
    return f"{stem}.png"


def combined_name(
    *,
    frame_index: int,
    mode: str = "source",
    source_path: Path | None = None,
    source_stem: str | None = None,
    prefix: str | None = None,
) -> str:
    """Return the filename for a combined diagnostic frame."""

    stem = frame_stem(
        frame_index=frame_index,
        mode=mode,
        source_path=source_path,
        source_stem=source_stem,
        prefix=prefix,
    )
    return f"{stem}_combined.png"


def masks_dir(output_dir: Path) -> Path:
    """Return the directory for clean mask frames."""

    return output_dir / "masks"


def combined_dir(output_dir: Path) -> Path:
    """Return the directory for combined diagnostic frames."""

    return output_dir / "combined"


def ensure_series_output_dirs(
    output_dir: Path,
    *,
    save_combined: bool = False,
) -> tuple[Path, Path | None]:
    """Create the standard output directories."""

    mask_output_dir = masks_dir(output_dir)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    combined_output_dir: Path | None = None

    if save_combined:
        combined_output_dir = combined_dir(output_dir)
        combined_output_dir.mkdir(parents=True, exist_ok=True)

    return mask_output_dir, combined_output_dir


def mask_frame_path(
    output_dir: Path,
    *,
    frame_index: int,
    mode: str = "source",
    source_path: Path | None = None,
    source_stem: str | None = None,
    prefix: str | None = None,
) -> Path:
    """Return the path for a clean mask frame."""

    return masks_dir(output_dir) / mask_name(
        frame_index=frame_index,
        mode=mode,
        source_path=source_path,
        source_stem=source_stem,
        prefix=prefix,
    )


def combined_frame_path(
    output_dir: Path,
    *,
    frame_index: int,
    mode: str = "source",
    source_path: Path | None = None,
    source_stem: str | None = None,
    prefix: str | None = None,
) -> Path:
    """Return the path for a combined diagnostic frame."""

    return combined_dir(output_dir) / combined_name(
        frame_index=frame_index,
        mode=mode,
        source_path=source_path,
        source_stem=source_stem,
        prefix=prefix,
    )


def combined_video_path(output_dir: Path) -> Path:
    """Return the path for the optional combined diagnostic MP4."""

    return output_dir / "combined_video.mp4"