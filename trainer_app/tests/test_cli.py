from vimsam_trainer.cli import main
from vimsam_trainer.core.result import TrainingResult


def test_main_builds_config_and_runs_app(monkeypatch, capsys, tmp_path):
    class FakeTrainerApp:
        def run(self, config):
            assert config.workflow == "raw_frames"
            assert config.patch_shape == (128, 256)
            assert config.with_segmentation_decoder is False
            return TrainingResult(
                success=True,
                message="ok",
                output_dir=config.output_path,
                train_count=1,
                val_count=1,
            )

    monkeypatch.setattr("vimsam_trainer.cli.TrainerApp", FakeTrainerApp)

    code = main(
        [
            "--images",
            str(tmp_path / "images"),
            "--masks",
            str(tmp_path / "masks"),
            "--out",
            str(tmp_path / "out"),
            "--patch-shape",
            "128,256",
            "--without-segmentation-decoder",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "ok" in captured.out
