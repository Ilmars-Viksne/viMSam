from .core.config import ModelConfig, PromptConfig, SegmentationResult, WorkflowConfig
from .core.errors import DependencyMissingError, InputValidationError, OutputWriteError, SegmenterError

__all__ = [
    "SegmenterApp",
    "WorkflowConfig",
    "PromptConfig",
    "ModelConfig",
    "SegmentationResult",
    "SegmenterError",
    "InputValidationError",
    "DependencyMissingError",
    "OutputWriteError",
]


def __getattr__(name: str):
    if name == "SegmenterApp":
        from .core.app import SegmenterApp

        return SegmenterApp
    raise AttributeError(name)
