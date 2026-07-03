from pathlib import Path

import pytest

from vimsam_segmenter.core.errors import InputValidationError
from vimsam_segmenter.utils.time_resolver import (
    resolve_time_seconds,
    time_seconds_from_filenames,
    validate_user_time_seconds,
)


def test_time_seconds_from_default_timestamp_filenames():
    paths = [
        Path("20260226213602.raw"),
        Path("20260226213625.raw"),
        Path("20260226213648.raw"),
    ]

    assert time_seconds_from_filenames(paths) == [0.0, 23.0, 46.0]


def test_resolve_time_seconds_prefers_user_values():
    paths = [Path("not-a-timestamp.raw"), Path("still-not.raw")]

    assert resolve_time_seconds(source_paths=paths, user_time_seconds=(0, 2.5)) == [0.0, 2.5]


def test_resolve_time_seconds_uses_fps_when_timestamp_disabled():
    paths = [Path("a.raw"), Path("b.raw"), Path("c.raw")]

    assert resolve_time_seconds(source_paths=paths, fps=2, timestamp_format=None) == [0.0, 0.5, 1.0]


def test_validate_user_time_seconds_rejects_bad_length():
    with pytest.raises(InputValidationError):
        validate_user_time_seconds((0.0, 1.0), 3)
