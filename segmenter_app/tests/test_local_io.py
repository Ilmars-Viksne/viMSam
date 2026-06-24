import pytest

from vimsam_segmenter.core.errors import OutputWriteError
from vimsam_segmenter.io.local import resolve_image_output, save_records


def test_resolve_image_output_rejects_file_as_directory(tmp_path):
    output = tmp_path / "existing"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OutputWriteError):
        resolve_image_output(output, tmp_path / "input.tif")


def test_save_records_writes_json_with_parent_creation(tmp_path):
    path = save_records(tmp_path / "nested" / "stats", [{"frame_id": 0, "area_px": 3}], "json")

    assert path == tmp_path / "nested" / "stats.json"
    assert path.exists()
