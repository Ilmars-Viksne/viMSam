from __future__ import annotations

from ..workflows import WORKFLOW_MAP
from .config import WorkflowConfig
from .model_service import ModelService


class SegmenterApp:
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
