from pathlib import Path

from vimsam_trainer.core.app import TrainerApp
from vimsam_trainer.core.config import TrainingConfig
from vimsam_trainer.core.result import TrainingResult


def test_trainer_app_collects_splits_and_calls_fine_tuner(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    for name in ("a", "b", "c"):
        (images / f"{name}.raw").write_bytes(b"")
        (masks / f"{name}.tif").write_bytes(b"mask")

    class FakeFineTuner:
        def run(self, *, config, train_pairs, val_pairs):
            assert config.workflow == "raw_frames"
            assert len(train_pairs) == 2
            assert len(val_pairs) == 1
            return TrainingResult(
                success=True,
                output_dir=Path(config.output_path),
                train_count=len(train_pairs),
                val_count=len(val_pairs),
            )

    result = TrainerApp(fine_tuner=FakeFineTuner()).run(
        TrainingConfig(
            workflow="raw_frames",
            images_path=images,
            masks_path=masks,
            output_path=tmp_path / "out",
            val_fraction=0.34,
        )
    )

    assert result.success
    assert result.train_count == 2
    assert result.val_count == 1
