import pytest

np = pytest.importorskip("numpy")

from vimsam_trainer.core.errors import InputValidationError
from vimsam_trainer.preprocessing.preprocess import PreProcessor


def test_preprocessor_matches_segmenter_fixed_16bit_behavior():
    image = np.array([0, 256, 65535], dtype=np.uint16)

    result = PreProcessor().run(image)

    assert result.dtype == np.uint8
    assert np.array_equal(result, np.array([0, 1, 255], dtype=np.uint8))


def test_preprocessor_supports_minmax():
    image = np.array([50, 100, 150], dtype=np.uint16)

    result = PreProcessor(method="minmax").run(image)

    assert result.dtype == np.uint8
    assert result[1] == 127


def test_preprocessor_rejects_invalid_method():
    with pytest.raises(InputValidationError):
        PreProcessor("invalid")
