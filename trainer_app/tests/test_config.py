from pathlib import Path

import pytest

from vimsam_trainer.core.config import TrainingConfig
from vimsam_trainer.core.errors import InputValidationError


def test_training_config_accepts_valid_raw_workflow(tmp_path):
    config = TrainingConfig(
        workflow="raw_frames",
        images_path=tmp_path / "images",
        masks_path=tmp_path / "masks",
        output_path=tmp_path / "out",
        raw_width=1024,
        raw_height=1024,
        patch_shape="256x512",
    )

    assert config.workflow == "raw_frames"
    assert config.model_type == "vit_b"
    assert config.patch_shape == (256, 512)
    assert isinstance(config.images_path, Path)


def test_training_config_rejects_invalid_workflow(tmp_path):
    with pytest.raises(InputValidationError):
        TrainingConfig(
            workflow="unknown",
            images_path=tmp_path / "images",
            masks_path=tmp_path / "masks",
            output_path=tmp_path / "out",
        )


def test_training_config_rejects_invalid_val_fraction(tmp_path):
    with pytest.raises(InputValidationError):
        TrainingConfig(
            workflow="raw_frames",
            images_path=tmp_path / "images",
            masks_path=tmp_path / "masks",
            output_path=tmp_path / "out",
            val_fraction=1.0,
        )


def test_training_config_rejects_invalid_patch_shape(tmp_path):
    with pytest.raises(InputValidationError):
        TrainingConfig(
            workflow="raw_frames",
            images_path=tmp_path / "images",
            masks_path=tmp_path / "masks",
            output_path=tmp_path / "out",
            patch_shape=(512, 0),
        )
