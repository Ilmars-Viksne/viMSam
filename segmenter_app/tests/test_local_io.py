import pytest
import pandas as pd

from vimsam_segmenter.core.errors import OutputWriteError
from vimsam_segmenter.io.local import resolve_image_output, save_records
from vimsam_segmenter.utils.stats_schema import STANDARD_STATS_COLUMNS


def test_resolve_image_output_rejects_file_as_directory(tmp_path):
    output = tmp_path / "existing"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OutputWriteError):
        resolve_image_output(output, tmp_path / "input.tif")


def test_save_records_writes_json_with_parent_creation(tmp_path):
    path = save_records(tmp_path / "nested" / "stats", [{"frame_id": 0, "area_px": 3}], "json")

    assert path == tmp_path / "nested" / "stats.json"
    assert path.exists()


def test_save_records_writes_standard_csv_columns_first(tmp_path):
    path = save_records(
        tmp_path / "stats",
        [{"area_px": 3, "frame_id": 0, "extra": "value"}],
        "csv",
    )

    df = pd.read_csv(path)
    assert list(df.columns[: len(STANDARD_STATS_COLUMNS)]) == STANDARD_STATS_COLUMNS
    assert df.loc[0, "extra"] == "value"
