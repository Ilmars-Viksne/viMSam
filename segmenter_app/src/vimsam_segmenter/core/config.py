from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import InputValidationError

WORKFLOWS = {
    "single",
    "video",
    "raw_single",
    "raw_timeseries",
    "image_frames_logits",
    "raw_timeseries_logits",
}
TRACKING_METHODS = {"box", "centroid", "pole"}
EXPORT_FORMATS = {"csv", "json"}
PREPROCESSING_METHODS = {"fixed_16bit", "minmax", "percentile", "none"}

FRAME_NAME_MODES = {
    "source",
    "prefix-source",
    "index",
    "prefix-index",
}

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
    save_combined_video: bool = False
    tracking_method: str = "box"
    export_format: str = "csv"
    preprocessing_method: str = "fixed_16bit"
    raw_width: int = 1024
    raw_height: int = 1024
    fps: float | None = None
    timestamp_format: str | None = "%Y%m%d%H%M%S"
    time_seconds: tuple[float, ...] | None = None
    frame_name_mode: str = "source"
    frame_name_prefix: str | None = None


    def __post_init__(self) -> None:
        if self.workflow not in WORKFLOWS:
            raise InputValidationError(
                f"workflow must be one of {sorted(WORKFLOWS)}, got {self.workflow!r}"
            )

        if self.tracking_method not in TRACKING_METHODS:
            raise InputValidationError(
                f"tracking_method must be one of {sorted(TRACKING_METHODS)}, "
                f"got {self.tracking_method!r}"
            )

        if self.export_format not in EXPORT_FORMATS:
            raise InputValidationError(
                f"export_format must be one of {sorted(EXPORT_FORMATS)}, "
                f"got {self.export_format!r}"
            )

        if self.frame_name_mode not in FRAME_NAME_MODES:
            allowed = ", ".join(sorted(FRAME_NAME_MODES))
            raise InputValidationError(
                f"frame_name_mode must be one of: {allowed}"
            )

        if self.frame_name_prefix is not None:
            self.frame_name_prefix = self.frame_name_prefix.strip()

            if not self.frame_name_prefix:
                self.frame_name_prefix = None

        if self.frame_name_mode in {"prefix-source", "prefix-index"}:
            if self.frame_name_prefix is None:
                raise InputValidationError(
                    f"frame_name_prefix is required when "
                    f"frame_name_mode='{self.frame_name_mode}'."
                )

        if self.frame_name_prefix is not None:
            self.frame_name_prefix = self.frame_name_prefix.strip()

            if not self.frame_name_prefix:
                self.frame_name_prefix = None
            elif "/" in self.frame_name_prefix or "\\" in self.frame_name_prefix:
                raise InputValidationError(
                    "frame_name_prefix must not contain path separators."
                )

        if self.frame_name_mode in {"prefix-source", "prefix-index"}:
            if self.frame_name_prefix is None:
                raise InputValidationError(
                    f"frame_name_prefix is required when "
                    f"frame_name_mode='{self.frame_name_mode}'."
                )

        self.preprocessing_method = self.preprocessing_method.strip().lower()
        if self.preprocessing_method not in PREPROCESSING_METHODS:
            raise InputValidationError(
                f"preprocessing_method must be one of {sorted(PREPROCESSING_METHODS)}, "
                f"got {self.preprocessing_method!r}"
            )

        if self.raw_width <= 0 or self.raw_height <= 0:
            raise InputValidationError("raw_width and raw_height must be positive")

        if self.fps is not None and self.fps <= 0:
            raise InputValidationError("fps must be positive")

        self.input_path = normalize_path(self.input_path)
        self.output_path = normalize_path(self.output_path)

        if self.prompts is None:
            self.prompts = PromptConfig()

        if self.timestamp_format == "":
            self.timestamp_format = None

        if self.time_seconds is not None:
            try:
                self.time_seconds = tuple(float(value) for value in self.time_seconds)
            except (TypeError, ValueError) as exc:
                raise InputValidationError("time_seconds must contain numeric values") from exc


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
