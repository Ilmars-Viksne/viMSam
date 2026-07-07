import sys
import types

import pytest

np = pytest.importorskip("numpy")
imageio = pytest.importorskip("imageio.v3")

from vimsam_trainer.core.config import TrainingConfig
from vimsam_trainer.data.pairs import ImageMaskPair
from vimsam_trainer.training.micro_sam_finetune import MicroSamFineTuner


def test_micro_sam_finetuner_passes_required_loader_and_save_root(monkeypatch, tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    for name in ("train", "val"):
        (images / f"{name}.raw").write_bytes(arr.tobytes())
        imageio.imwrite(masks / f"{name}.tif", np.array([[0, 1], [2, 3]], dtype=np.uint16))

    calls = {"loaders": [], "train": None}

    def fake_default_sam_loader(**kwargs):
        calls["loaders"].append(kwargs)
        return {"loader": len(calls["loaders"])}

    def fake_train_sam(**kwargs):
        calls["train"] = kwargs
        checkpoint_dir = tmp_path / "out" / "checkpoints" / "out"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")

    training_module = types.ModuleType("micro_sam.training")
    training_module.default_sam_loader = fake_default_sam_loader
    training_module.train_sam = fake_train_sam
    micro_sam_module = types.ModuleType("micro_sam")
    micro_sam_module.training = training_module
    monkeypatch.setitem(sys.modules, "micro_sam", micro_sam_module)
    monkeypatch.setitem(sys.modules, "micro_sam.training", training_module)

    config = TrainingConfig(
        workflow="raw_frames",
        images_path=images,
        masks_path=masks,
        output_path=tmp_path / "out",
        raw_width=4,
        raw_height=4,
        patch_shape=(4, 4),
        epochs=1,
        batch_size=2,
    )

    result = MicroSamFineTuner().run(
        config=config,
        train_pairs=[ImageMaskPair(images / "train.raw", masks / "train.tif")],
        val_pairs=[ImageMaskPair(images / "val.raw", masks / "val.tif")],
    )

    assert result.success
    assert result.checkpoint_path == tmp_path / "out" / "checkpoints" / "out" / "best.pt"
    assert calls["loaders"][0]["raw_key"] is None
    assert calls["loaders"][0]["label_key"] is None
    assert calls["loaders"][0]["patch_shape"] == (4, 4)
    assert calls["loaders"][0]["with_segmentation_decoder"] is True
    assert calls["train"]["save_root"] == str(tmp_path / "out")
