import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vimsam_segmenter.cli import parse_box, parse_points
from vimsam_segmenter.core.config import ModelConfig
from vimsam_segmenter.core.errors import InputValidationError, DependencyMissingError, OutputWriteError
from vimsam_segmenter.io.raw import validate_raw_file_size
from vimsam_segmenter.io.local import resolve_image_output, save_records


def test_parse_points_raises_argparse_for_bad_format():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_points("500")


def test_parse_box_rejects_non_increasing_coords():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_box("100,150,100,200")


def test_modelconfig_validates_device_and_normalizes_checkpoint(tmp_path):
    with pytest.raises(InputValidationError):
        ModelConfig(model_type="", device="auto")

    with pytest.raises(InputValidationError):
        ModelConfig(model_type="vit_b", device="invalid")

    cfg = ModelConfig(model_type="vit_b", device="cpu", checkpoint_path=tmp_path / "ckpt.pth")
    assert isinstance(cfg.checkpoint_path, Path)


def test_validate_raw_file_size_rejects_non_integer_dimensions(tmp_path):
    path = tmp_path / "f.raw"
    data = np.arange(4, dtype=np.uint16).reshape(2, 2)
    path.write_bytes(data.tobytes())

    with pytest.raises(InputValidationError):
        validate_raw_file_size(path, width="x", height=2)


def test_resolve_image_output_variants(tmp_path):
    input_path = tmp_path / "input.tif"
    input_path.write_text("x")

    # directory exists
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    p = resolve_image_output(out_dir, input_path)
    assert p.parent == out_dir
    assert p.suffix == ".png"
    assert p.name.startswith("res_input")

    # explicit file with suffix (parent created)
    explicit = tmp_path / "nested" / "file.png"
    p2 = resolve_image_output(explicit, input_path)
    assert p2 == explicit

    # path without suffix should create directory and default name
    new_dir = tmp_path / "newout"
    p3 = resolve_image_output(new_dir, input_path)
    assert p3.parent == new_dir
    assert p3.name.startswith("res_input")


def test_save_records_csv_and_json_and_invalid_format(tmp_path, monkeypatch):
    records = [{"frame_id": 0, "area_px": 3}]

    # JSON should succeed without pandas
    path_json = save_records(tmp_path / "stats_json", records, "json")
    assert path_json is not None
    assert path_json.exists()

    # CSV requires pandas — provide a fake pandas
    fake_pd = types.SimpleNamespace()

    class FakeDF:
        def __init__(self, records):
            self._records = records

        def to_csv(self, handle, index=False):
            handle.write("frame_id,area_px\n")
            for r in self._records:
                handle.write(f"{r['frame_id']},{r['area_px']}\n")

    fake_pd.DataFrame = FakeDF
    monkeypatch.setitem(sys.modules, "pandas", fake_pd)

    path_csv = save_records(tmp_path / "stats_csv", records, "csv")
    assert path_csv is not None
    assert path_csv.exists()

    # invalid format
    with pytest.raises(InputValidationError):
        save_records(tmp_path / "stats_bad", records, "xml")
