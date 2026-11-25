import sys
import types
from pathlib import Path

import pytest

from vimsam_segmenter.core.errors import DependencyMissingError
from vimsam_segmenter.core.model_service import ModelService


def test_get_predictor_reports_missing_ml_dependencies(monkeypatch):
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        if name == "micro_sam" or name.startswith("micro_sam"):
            raise ImportError("No module named 'micro_sam'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    service = ModelService()
    with pytest.raises(DependencyMissingError, match="ML dependencies are missing"):
        service.get_predictor()


def test_get_predictor_rejects_cuda_when_unavailable(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_micro_sam = types.ModuleType("micro_sam")
    fake_util = types.ModuleType("micro_sam.util")
    fake_util.get_sam_model = lambda **kwargs: kwargs
    fake_micro_sam.util = fake_util

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "micro_sam", fake_micro_sam)
    monkeypatch.setitem(sys.modules, "micro_sam.util", fake_util)

    service = ModelService(device="cuda", checkpoint_path=Path("/tmp/model.pth"))
    with pytest.raises(DependencyMissingError, match="CUDA was requested but is not available"):
        service.get_predictor()


def test_get_predictor_uses_auto_device_when_cuda_available(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)

    fake_micro_sam = types.ModuleType("micro_sam")
    fake_util = types.ModuleType("micro_sam.util")
    fake_util.get_sam_model = lambda **kwargs: kwargs
    fake_micro_sam.util = fake_util

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "micro_sam", fake_micro_sam)
    monkeypatch.setitem(sys.modules, "micro_sam.util", fake_util)

    service = ModelService(device="auto")
    predictor = service.get_predictor()

    assert predictor["device"] == "cuda"
    assert predictor["model_type"] == "vit_b"
