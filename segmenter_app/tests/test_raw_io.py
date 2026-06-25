import numpy as np
import pytest

from vimsam_segmenter.core.errors import InputValidationError
from vimsam_segmenter.io.raw import read_u3cmos_raw, validate_raw_file_size


def test_read_raw_validates_and_returns_uint8(tmp_path):
    path = tmp_path / "frame.raw"
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)
    path.write_bytes(data.tobytes())

    validate_raw_file_size(path, width=4, height=4)
    image = read_u3cmos_raw(path, width=4, height=4)

    assert image.shape == (4, 4)
    assert image.dtype == np.uint8


def test_read_raw_rejects_invalid_size(tmp_path):
    path = tmp_path / "bad.raw"
    path.write_bytes(b"1234")

    with pytest.raises(InputValidationError, match="Raw file size mismatch"):
        read_u3cmos_raw(path, width=4, height=4)


def test_read_raw_rejects_too_large_size(tmp_path):
    path = tmp_path / "too_large.raw"
    path.write_bytes(np.arange(17, dtype=np.uint16).tobytes())

    with pytest.raises(InputValidationError, match="Raw file size mismatch"):
        read_u3cmos_raw(path, width=4, height=4)
