from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def split_pairs(
    pairs: Sequence[T],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[T], list[T]]:
    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)

    val_count = max(1, int(round(len(items) * val_fraction)))
    if val_count >= len(items):
        val_count = len(items) - 1

    return items[val_count:], items[:val_count]
