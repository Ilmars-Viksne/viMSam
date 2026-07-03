from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..core.errors import InputValidationError


DEFAULT_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


def _parse_datetime_from_stem(path: Path, timestamp_format: str) -> datetime:
    try:
        return datetime.strptime(path.stem, timestamp_format)
    except ValueError as exc:
        raise InputValidationError(
            f"Could not parse timestamp from filename '{path.name}' "
            f"using format '{timestamp_format}'."
        ) from exc


def time_seconds_from_filenames(
    paths: Sequence[Path],
    *,
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
) -> list[float]:
    if not paths:
        return []

    timestamps = [_parse_datetime_from_stem(path, timestamp_format) for path in paths]
    start = timestamps[0]
    return [float((timestamp - start).total_seconds()) for timestamp in timestamps]


def time_seconds_from_fps(frame_count: int, fps: float) -> list[float]:
    if fps <= 0:
        raise InputValidationError("fps must be positive.")
    return [frame_id / fps for frame_id in range(frame_count)]


def validate_user_time_seconds(values: Sequence[float], frame_count: int) -> list[float]:
    if len(values) != frame_count:
        raise InputValidationError(f"Expected {frame_count} time values, got {len(values)}.")

    result = [float(value) for value in values]
    if result and result[0] != 0:
        raise InputValidationError("The first user-provided time value must be 0 seconds.")

    for previous, current in zip(result, result[1:]):
        if current < previous:
            raise InputValidationError("User-provided time values must be non-decreasing.")

    return result


def resolve_time_seconds(
    *,
    source_paths: Sequence[Path],
    fps: float | None = None,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
    user_time_seconds: Sequence[float] | None = None,
) -> list[float]:
    frame_count = len(source_paths)

    if user_time_seconds is not None:
        return validate_user_time_seconds(user_time_seconds, frame_count)

    if timestamp_format:
        return time_seconds_from_filenames(source_paths, timestamp_format=timestamp_format)

    if fps is not None:
        return time_seconds_from_fps(frame_count, fps)

    return [float(frame_id) for frame_id in range(frame_count)]
