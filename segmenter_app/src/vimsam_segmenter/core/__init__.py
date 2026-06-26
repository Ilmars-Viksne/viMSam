from .config import ModelConfig, PromptConfig, SegmentationResult, WorkflowConfig
from .errors import DependencyMissingError, InputValidationError, OutputWriteError, SegmenterError

__all__ = [
    "SegmenterApp",
    "WorkflowConfig",
    "PromptConfig",
    "ModelConfig",
    "SegmentationResult",
    "ModelService",
    "SegmenterError",
    "InputValidationError",
    "DependencyMissingError",
    "OutputWriteError",
]


def __getattr__(name: str):
    if name == "SegmenterApp":
        from .app import SegmenterApp

        return SegmenterApp
    if name == "ModelService":
        from .model_service import ModelService

        return ModelService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
