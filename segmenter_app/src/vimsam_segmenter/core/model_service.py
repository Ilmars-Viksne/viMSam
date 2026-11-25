from __future__ import annotations

from pathlib import Path

from .errors import DependencyMissingError
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelService:
    def __init__(self, model_type: str = "vit_b", device: str = "auto", checkpoint_path: Path | None = None) -> None:
        self.model_type = model_type
        self.requested_device = device
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.device: str | None = None

    def _import_ml_dependencies(self) -> tuple[object, object]:
        try:
            import torch
        except ImportError as exc:
            raise DependencyMissingError(
                "ML dependencies are missing. Install with: pip install -e '.[ml]' or use the Colab mamba environment."
            ) from exc

        try:
            import micro_sam.util as micro_sam_util
        except ImportError as exc:
            raise DependencyMissingError(
                "ML dependencies are missing. Install with: pip install -e '.[ml]' or use the Colab mamba environment."
            ) from exc

        return torch, micro_sam_util

    def _resolve_device(self, torch_module: object) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if self.requested_device == "cuda" and not torch_module.cuda.is_available():
            raise DependencyMissingError("CUDA was requested but is not available.")
        return self.requested_device

    def load_model(self, model_type: str | None = None) -> None:
        if self.predictor is not None:
            return
        if model_type is not None:
            self.model_type = model_type

        torch_module, micro_sam_util = self._import_ml_dependencies()
        self.device = self._resolve_device(torch_module)

        kwargs = {"model_type": self.model_type, "device": self.device}
        if self.checkpoint_path is not None:
            kwargs["checkpoint_path"] = str(self.checkpoint_path)
        logger.info("Loading MicroSAM (%s) on %s...", self.model_type, self.device)

        try:
            self.predictor = micro_sam_util.get_sam_model(return_sam=False, **kwargs)
        except TypeError as exc:
            if "return_sam" in str(exc):
                self.predictor = micro_sam_util.get_sam_model(**kwargs)
            else:
                raise

    def get_predictor(self):
        if self.predictor is None:
            self.load_model()
        return self.predictor
