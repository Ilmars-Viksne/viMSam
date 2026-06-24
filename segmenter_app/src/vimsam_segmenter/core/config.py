from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import InputValidationError

WORKFLOWS = {"single", "video", "raw_single", "raw_timeseries"}
TRACKING_METHODS = {"box", "centroid", "pole"}
EXPORT_FORMATS = {"csv", "json"}


def normalize_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


@dataclass(slots=True)
class ModelConfig:
    model_type: str = "vit_b"
    device: str = "auto"
    checkpoint_path: Path | None = None

    @property
    def name(self) -> str:
        return self.model_type

    def __post_init__(self) -> None:
        self.model_type = self.model_type.strip()
        if not self.model_type:
            raise InputValidationError("model type must not be empty")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise InputValidationError("device must be one of: auto, cpu, cuda")
        if self.checkpoint_path is not None:
            self.checkpoint_path = normalize_path(self.checkpoint_path)


@dataclass(slots=True)
class PromptConfig:
    points: tuple[tuple[int, int], ...] | None = None
    box: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.points is not None:
            try:
                self.points = tuple((int(x), int(y)) for x, y in self.points)
            except (TypeError, ValueError) as exc:
                raise InputValidationError("Prompt points must be iterable (x, y) integer pairs") from exc
        if self.box is not None:
            try:
                self.box = tuple(int(v) for v in self.box)
            except (TypeError, ValueError) as exc:
                raise InputValidationError("Prompt box must contain exactly four integers") from exc
            if len(self.box) != 4:
                raise InputValidationError("Prompt box must contain exactly four integers")


@dataclass(slots=True)
class WorkflowConfig:
    workflow: str
    input_path: Path | str
    output_path: Path | str
    model: ModelConfig = field(default_factory=ModelConfig)
    prompts: PromptConfig | None = None
    show_prompts: bool = False
    save_combined: bool = False
    tracking_method: str = "box"
    export_format: str = "csv"
    raw_width: int = 1024
    raw_height: int = 1024

    def __post_init__(self) -> None:
        self.workflow = self.workflow.strip()
        if self.workflow not in WORKFLOWS:
            raise InputValidationError(f"workflow must be one of: {', '.join(sorted(WORKFLOWS))}")
        if self.tracking_method not in TRACKING_METHODS:
            raise InputValidationError("tracking_method must be one of: box, centroid, pole")
        if self.export_format not in EXPORT_FORMATS:
            raise InputValidationError("format must be one of: csv, json")
        self.input_path = normalize_path(self.input_path)
        self.output_path = normalize_path(self.output_path)
        if not isinstance(self.model, ModelConfig):
            raise InputValidationError("model must be a ModelConfig")
        if not isinstance(self.raw_width, int) or not isinstance(self.raw_height, int):
            raise InputValidationError("raw dimensions must be integers")
        if self.raw_width <= 0 or self.raw_height <= 0:
            raise InputValidationError("raw dimensions must be positive integers")

    @property
    def workflow_type(self) -> str:
        return self.workflow

    @property
    def input_uri(self) -> str:
        return str(self.input_path)

    @property
    def output_uri(self) -> str:
        return str(self.output_path)


@dataclass(slots=True)
class SegmentationResult:
    success: bool
    count: int = 0
    message: str = ""
    outputs: tuple[Path, ...] = ()
    stats_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
