from __future__ import annotations

"""Application orchestration for vimsam_segmenter.

SegmenterApp.run() is the primary public API for the project.
It validates and dispatches workflows based on WorkflowConfig,
and it is the shared entry point for the CLI, Python API,
notebooks, and tests.
"""

from ..workflows import WORKFLOW_MAP
from .config import WorkflowConfig
from .model_service import ModelService


class SegmenterApp:
    """Main application orchestrator.

    Public API:
    - CLI: `vimsam-segmenter`
    - Python API: `SegmenterApp().run(config)`
    - Notebook and tests can also call this shared entry point.
    """
    def __init__(self, model_service: ModelService | None = None) -> None:
        self.model_service = model_service

    def run(self, config: WorkflowConfig):
        service = self.model_service or ModelService(
            model_type=config.model.model_type,
            device=config.model.device,
            checkpoint_path=config.model.checkpoint_path,
        )
        workflow_class = WORKFLOW_MAP[config.workflow]
        return workflow_class(service).run(config)
