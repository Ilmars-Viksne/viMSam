from __future__ import annotations

import numpy as np


def build_prompt_overlay(
    *,
    points: tuple[tuple[int, int], ...] | None = None,
    box: tuple[int, int, int, int] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Build a normalized prompt dictionary for visualization.

    Coordinate convention:
        points: (x, y)
        box:    (x1, y1, x2, y2)
    """
    prompts: dict[str, np.ndarray] = {}

    if points:
        prompts["points"] = np.asarray(points, dtype=np.int32)

    if box is not None:
        prompts["box"] = np.asarray(box, dtype=np.int32).reshape(4)

    return prompts
