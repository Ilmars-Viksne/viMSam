from __future__ import annotations

import numpy as np

from vimsam_segmenter.core.config import SegmentationResult, WorkflowConfig


class BaseWorkflow:
    def __init__(self, model_service) -> None:
        self.model_service = model_service

    def run(self, config: WorkflowConfig) -> SegmentationResult:
        raise NotImplementedError

    @staticmethod
    def sam_image(processed: np.ndarray) -> np.ndarray:
        return np.stack((processed,) * 3, axis=-1) if processed.ndim == 2 else processed


def automatic_mask_generator(predictor):
    try:
        from micro_sam.instance_segmentation import AutomaticMaskGenerator
    except ImportError as exc:
        from vimsam_segmenter.core.errors import DependencyMissingError

        raise DependencyMissingError("micro_sam is required for automatic segmentation.") from exc
    return AutomaticMaskGenerator(predictor)
