from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingResult:
    success: bool
    count: int = 0
    message: str = ""
    output_dir: Path | None = None
    checkpoint_path: Path | None = None
    summary_path: Path | None = None
    train_count: int = 0
    val_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
