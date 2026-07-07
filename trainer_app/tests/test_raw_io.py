import pytest

np = pytest.importorskip("numpy")

from vimsam_trainer.core.errors import InputValidationError
from vimsam_trainer.data.raw_io import read_u16_raw


def test_read_u16_raw_validates_and_reads(tmp_path):
    path = tmp_path / "frame.raw"
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    path.write_bytes(arr.tobytes())

    result = read_u16_raw(path, width=4, height=4)

    assert result.shape == (4, 4)
    assert result.dtype == np.uint16
    assert np.array_equal(result, arr)


def test_read_u16_raw_rejects_wrong_size(tmp_path):
    path = tmp_path / "bad.raw"
    path.write_bytes(b"1234")

    with pytest.raises(InputValidationError):
        read_u16_raw(path, width=4, height=4)
