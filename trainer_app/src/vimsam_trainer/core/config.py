from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import InputValidationError

WORKFLOWS = {
    "raw_frames",
    "image_frames",
}

PREPROCESSING_METHODS = {
    "fixed_16bit",
    "minmax",
    "percentile",
    "none",
}

DEVICES = {
    "auto",
    "cpu",
    "cuda",
}


def normalize_path(path: Path | str) -> Path:
    """Normalize a filesystem path without requiring it to exist."""
    return Path(path).expanduser().resolve(strict=False)


def _normalize_patch_shape(
    value: tuple[int, int] | list[int] | str,
) -> tuple[int, int]:
    """
    Normalize a patch-shape value to a two-integer tuple.

    Accepted inputs include:

    - ``(512, 512)``
    - ``[512, 512]``
    - ``"512,512"``
    - ``"512x512"``
    """
    if isinstance(value, str):
        parts = [
            part.strip()
            for part in value.replace("x", ",").split(",")
        ]

        if len(parts) != 2 or any(not part for part in parts):
            raise InputValidationError(
                "patch_shape must contain width,height, "
                "for example 512,512"
            )

        try:
            width, height = (
                int(part)
                for part in parts
            )
        except ValueError as exc:
            raise InputValidationError(
                "patch_shape must contain integer values"
            ) from exc

    else:
        try:
            width, height = (
                int(part)
                for part in value
            )
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "patch_shape must contain exactly two integers"
            ) from exc

    if width <= 0 or height <= 0:
        raise InputValidationError(
            "patch_shape values must be positive"
        )

    return width, height


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for a viMSam fine-tuning run."""

    workflow: str
    images_path: Path | str
    masks_path: Path | str
    output_path: Path | str

    model_type: str = "vit_b"
    device: str = "auto"
    checkpoint_path: Path | str | None = None

    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-5
    val_fraction: float = 0.2
    seed: int = 42

    raw_width: int = 1024
    raw_height: int = 1024
    preprocessing_method: str = "fixed_16bit"

    patch_shape: tuple[int, int] | list[int] | str = (
        512,
        512,
    )

    num_workers: int = 0
    n_objects_per_batch: int = 25
    with_segmentation_decoder: bool = True
    min_size: int = 25
    min_instances_per_patch: int = 1

    def __post_init__(self) -> None:
        """Normalize and validate all configuration values."""
        self.workflow = self.workflow.strip().lower()

        if self.workflow not in WORKFLOWS:
            raise InputValidationError(
                "workflow must be one of "
                f"{sorted(WORKFLOWS)}, "
                f"got {self.workflow!r}"
            )

        self.preprocessing_method = (
            self.preprocessing_method.strip().lower()
        )

        if self.preprocessing_method not in PREPROCESSING_METHODS:
            raise InputValidationError(
                "preprocessing_method must be one of "
                f"{sorted(PREPROCESSING_METHODS)}, "
                f"got {self.preprocessing_method!r}"
            )

        self.device = self.device.strip().lower()

        if self.device not in DEVICES:
            raise InputValidationError(
                "device must be one of "
                f"{sorted(DEVICES)}, "
                f"got {self.device!r}"
            )

        self.model_type = self.model_type.strip()

        if not self.model_type:
            raise InputValidationError(
                "model_type must not be empty"
            )

        if self.epochs <= 0:
            raise InputValidationError(
                "epochs must be positive"
            )

        if self.batch_size <= 0:
            raise InputValidationError(
                "batch_size must be positive"
            )

        if self.learning_rate <= 0:
            raise InputValidationError(
                "learning_rate must be positive"
            )

        if not 0.0 < self.val_fraction < 1.0:
            raise InputValidationError(
                "val_fraction must be between 0 and 1"
            )

        if self.raw_width <= 0 or self.raw_height <= 0:
            raise InputValidationError(
                "raw_width and raw_height must be positive"
            )

        if self.num_workers < 0:
            raise InputValidationError(
                "num_workers must be non-negative"
            )

        if self.n_objects_per_batch <= 0:
            raise InputValidationError(
                "n_objects_per_batch must be positive"
            )

        if self.min_size <= 0:
            raise InputValidationError(
                "min_size must be positive"
            )

        if self.min_instances_per_patch <= 0:
            raise InputValidationError(
                "min_instances_per_patch must be positive"
            )

        self.patch_shape = _normalize_patch_shape(
            self.patch_shape
        )

        self.images_path = normalize_path(
            self.images_path
        )
        self.masks_path = normalize_path(
            self.masks_path
        )
        self.output_path = normalize_path(
            self.output_path
        )

        if self.checkpoint_path is not None:
            self.checkpoint_path = normalize_path(
                self.checkpoint_path
            )