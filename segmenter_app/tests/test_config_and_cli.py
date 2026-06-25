from pathlib import Path

import pytest

from src.cli import parse_box, parse_points
from src.core.config import PromptConfig, WorkflowConfig
from src.core.errors import InputValidationError


def test_parse_points_preserves_order():
    assert parse_points("500,480 150,100") == ((500, 480), (150, 100))


@pytest.mark.parametrize("value", ["500", "x,2", "1,2,3"])
def test_parse_points_rejects_invalid_formats(value):
    with pytest.raises(Exception):
        parse_points(value)


def test_parse_points_returns_none_for_empty_input():
    assert parse_points(None) is None
    assert parse_points("") is None
    assert parse_points("   ") is None


def test_parse_box_accepts_valid_box():
    assert parse_box("100,150,300,400") == (100, 150, 300, 400)


def test_parse_box_accepts_whitespace_around_values():
    assert parse_box(" 100, 150, 300, 400 ") == (100, 150, 300, 400)


def test_parse_box_returns_none_for_empty_input():
    assert parse_box(None) is None
    assert parse_box("") is None
    assert parse_box("   ") is None


@pytest.mark.parametrize(
    "value",
    [
        "100",
        "100,150",
        "100,150,300",
        "100,150,300,400,500",
        "100,abc,300,400",
        "100,150,x,400",
        "100,150,100,400",
        "100,150,300,150",
        "100,150,99,400",
        "100,150,300,149",
    ],
)
def test_parse_box_rejects_invalid_values(value):
    with pytest.raises(Exception):
        parse_box(value)


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

    assert config.workflow == "raw_single"
    assert isinstance(config.input_path, Path)
    assert isinstance(config.output_path, Path)
    assert config.prompts is not None
    assert config.prompts.points == ((1, 2),)
    assert config.tracking_method == "pole"
    assert config.raw_width == 32
    assert config.raw_height == 16


def test_workflow_config_preserves_box_prompt(tmp_path):
    config = WorkflowConfig(
        workflow="single",
        input_path="input.tif",
        output_path=tmp_path / "out.png",
        prompts=PromptConfig(box=(10, 20, 100, 200)),
    )

    assert config.prompts is not None
    assert config.prompts.box == (10, 20, 100, 200)


def test_workflow_config_preserves_points_and_box_prompts(tmp_path):
    config = WorkflowConfig(
        workflow="single",
        input_path="input.tif",
        output_path=tmp_path / "out.png",
        prompts=PromptConfig(
            points=((50, 60),),
            box=(10, 20, 100, 200),
        ),
    )

    assert config.prompts is not None
    assert config.prompts.points == ((50, 60),)
    assert config.prompts.box == (10, 20, 100, 200)


def test_workflow_config_rejects_invalid_values(tmp_path):
    with pytest.raises(InputValidationError):
        WorkflowConfig(
            workflow="unknown",
            input_path="in",
            output_path=tmp_path,
        )

    with pytest.raises(InputValidationError):
        WorkflowConfig(
            workflow="single",
            input_path="in",
            output_path=tmp_path,
            tracking_method="mean",
        )

    with pytest.raises(InputValidationError):
        WorkflowConfig(
            workflow="single",
            input_path="in",
            output_path=tmp_path,
            raw_width=0,
        )

    with pytest.raises(InputValidationError):
        WorkflowConfig(
            workflow="single",
            input_path="in",
            output_path=tmp_path,
            raw_height=0,
        )


def test_workflow_config_rejects_invalid_export_format(tmp_path):
    with pytest.raises(InputValidationError):
        WorkflowConfig(
            workflow="single",
            input_path="in",
            output_path=tmp_path,
            export_format="xlsx",
        )