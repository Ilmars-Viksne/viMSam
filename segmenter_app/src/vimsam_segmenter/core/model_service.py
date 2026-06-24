from __future__ import annotations

from pathlib import Path

from .errors import DependencyMissingError
from vimsam_segmenter.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelService:
    def __init__(self, model_type: str = "vit_b", device: str = "auto", checkpoint_path: Path | None = None) -> None:
        self.model_type = model_type
        self.requested_device = device
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.device: str | None = None

    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        try:
            import torch
        except ImportError as exc:
            raise DependencyMissingError("PyTorch is required to auto-detect CUDA. Install the ML extra or pass --device cpu.") from exc
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type: str | None = None) -> None:
        if self.predictor is not None:
            return
        if model_type is not None:
            self.model_type = model_type
        try:
            import micro_sam.util
        except ImportError as exc:
            raise DependencyMissingError("micro_sam is required for segmentation workflows. Install the ML dependencies first.") from exc

        self.device = self._resolve_device()
        kwargs = {"model_type": self.model_type, "device": self.device}
        if self.checkpoint_path is not None:
            kwargs["checkpoint_path"] = str(self.checkpoint_path)
        logger.info("Loading MicroSAM (%s) on %s...", self.model_type, self.device)
        self.predictor = micro_sam.util.get_sam_model(**kwargs)

    def get_predictor(self):
        if self.predictor is None:
            self.load_model()
        return self.predictor
