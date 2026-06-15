from abc import ABC, abstractmethod
from src.core.config import WorkflowConfig, SegmentationResult
from src.core.io import IOFactory

class BaseWorkflow(ABC):
    def __init__(self, model_service):
        self.model_service = model_service

    @abstractmethod
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        pass
