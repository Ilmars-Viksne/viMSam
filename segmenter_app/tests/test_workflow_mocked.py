import numpy as np
from pathlib import Path

from vimsam_segmenter.core.app import SegmenterApp
from vimsam_segmenter.core.config import WorkflowConfig, PromptConfig


class FakePredictor:
    def __init__(self):
        self.image = None

    def set_image(self, image):
        self.image = image

    def predict(self, *args, **kwargs):
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        # produce a small square mask in the image center-ish
        mask[2:5, 2:5] = True
        return [mask], [1.0], None


class FakeModelService:
    def get_predictor(self):
        return FakePredictor()


def make_raw_file(path: Path, width: int, height: int):
    arr = np.arange(width * height, dtype=np.uint16).reshape((height, width))
    path.write_bytes(arr.tobytes())


def test_raw_single_workflow_with_fake_predictor(tmp_path, monkeypatch):
    input_path = tmp_path / "frame.raw"
    make_raw_file(input_path, width=8, height=8)

    out_path = tmp_path / "out.png"

    # avoid requiring imageio during tests
    monkeypatch.setattr("vimsam_segmenter.io.local.save_image", lambda path, image: path)

    config = WorkflowConfig(
        workflow="raw_single",
        input_path=input_path,
        output_path=out_path,
        prompts=PromptConfig(points=((3, 3),)),
        raw_width=8,
        raw_height=8,
        export_format="json",
    )

    app = SegmenterApp(model_service=FakeModelService())
    result = app.run(config)

    assert result.success is True
    assert result.outputs
    # outputs should be paths (we stubbed save_image to return the path)
    for p in result.outputs:
        assert isinstance(p, Path)
    # stats file should be written when prompts are provided
    assert result.stats_path is not None
    assert result.stats_path.exists()
