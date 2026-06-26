import numpy as np
import pytest

from vimsam_segmenter.core.config import WorkflowConfig
from vimsam_segmenter.core.errors import InputValidationError
from vimsam_segmenter.processing.preprocess import PreProcessor


def test_preprocessor_uses_fixed_16bit_by_default():
    image = np.array([0, 256, 65535], dtype=np.uint16)

    result = PreProcessor().run(image)

    assert result.dtype == np.uint8
    assert np.array_equal(result, np.array([0, 1, 255], dtype=np.uint8))


def test_preprocessor_supports_percentile_scaling():
    image = np.array([0, 100, 200, 1000], dtype=np.uint16)

    result = PreProcessor(method="percentile").run(image)

    assert result.dtype == np.uint8
    assert result[0] == 0
    assert result[-1] == 255


def test_preprocessor_supports_minmax_scaling():
    image = np.array([50, 100, 150], dtype=np.uint16)

    result = PreProcessor(method="minmax").run(image)

    assert result.dtype == np.uint8
    assert result[1] == 127


def test_preprocessor_none_leaves_dtype_unchanged():
    image = np.array([100, 200], dtype=np.uint16)

    result = PreProcessor(method="none").run(image)

    assert result.dtype == np.uint16
    assert np.array_equal(result, image)


def test_workflow_config_rejects_invalid_preprocessing_method(tmp_path):
    with pytest.raises(InputValidationError):
        WorkflowConfig(
            workflow="single",
            input_path="in",
            output_path=tmp_path,
            preprocessing_method="invalid",
        )
