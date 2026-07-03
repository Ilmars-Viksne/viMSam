from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .local import save_video
from .series_outputs import combined_video_path


def save_combined_video(
    output_dir: Path,
    frames: Iterable[np.ndarray],
    fps: float | int | None = None,
) -> Path:
    """Save combined visualization frames as an MP4 video.

    This helper writes only combined/diagnostic frames.
    It never writes mask-only video.
    """
    video_fps = fps or 5
    return save_video(
        combined_video_path(output_dir),
        frames,
        fps=int(video_fps),
    )
