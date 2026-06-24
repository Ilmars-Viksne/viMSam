from pathlib import Path

import pytest

from src.cli import parse_points
from src.core.config import PromptConfig, WorkflowConfig
from src.core.errors import InputValidationError


def test_parse_points_preserves_order():
    assert parse_points("500,480 150,100") == ((500, 480), (150, 100))


@pytest.mark.parametrize("value", ["500", "x,2", "1,2,3"])
def test_parse_points_rejects_invalid_formats(value):
    with pytest.raises(Exception):
        parse_points(value)


def test_workflow_config_normalizes_paths_and_options(tmp_path):
    config = WorkflowConfig(
        workflow="raw_single",
        input_path="input.raw",
        output_path=tmp_path / "out.png",
        prompts=PromptConfig(points=((1, 2),)),
        tracking_method="pole",
        raw_width=32,
        raw_height=16,
    )

    assert config.input_path == Path("input.raw").resolve()
    assert config.output_path == tmp_path / "out.png"
    assert config.prompts.points == ((1, 2),)
    assert config.workflow_type == "raw_single"


def test_workflow_config_rejects_invalid_values(tmp_path):
    with pytest.raises(InputValidationError):
        WorkflowConfig(workflow="unknown", input_path="in", output_path=tmp_path)
    with pytest.raises(InputValidationError):
        WorkflowConfig(workflow="single", input_path="in", output_path=tmp_path, tracking_method="mean")
    with pytest.raises(InputValidationError):
        WorkflowConfig(workflow="single", input_path="in", output_path=tmp_path, raw_width=0)
