from pathlib import Path

import pytest

from vimsam_segmenter.cli import main, parse_box, parse_points, parse_time_seconds
from vimsam_segmenter.core.config import PromptConfig, SegmentationResult, WorkflowConfig
from vimsam_segmenter.core.errors import InputValidationError, SegmenterError


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


def test_parse_time_seconds_accepts_comma_separated_values():
    assert parse_time_seconds("0,23,46") == (0.0, 23.0, 46.0)


@pytest.mark.parametrize("value", ["0,,2", "0,x", ",1", "1,"])
def test_parse_time_seconds_rejects_invalid_values(value):
    with pytest.raises(Exception):
        parse_time_seconds(value)


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


def test_workflow_config_accepts_logits_workflows_and_fps(tmp_path):
    config = WorkflowConfig(
        workflow="image_frames_logits",
        input_path="frames",
        output_path=tmp_path / "out",
        fps=10.5,
    )

    assert config.workflow == "image_frames_logits"
    assert config.fps == 10.5


def test_workflow_config_does_not_save_combined_video_by_default(tmp_path):
    config = WorkflowConfig(
        workflow="raw_timeseries_logits",
        input_path=tmp_path / "input",
        output_path=tmp_path / "output",
    )

    assert config.save_combined_video is False


def test_main_passes_fps_to_workflow_config(monkeypatch, capsys, tmp_path):
    output_path = tmp_path / "result.png"

    class DummyApp:
        def run(self, config):
            assert config.workflow == "raw_timeseries_logits"
            assert config.fps == 12.5
            assert config.timestamp_format is None
            assert config.time_seconds == (0.0, 0.5)
            return SegmentationResult(success=True, outputs=(output_path,))

    monkeypatch.setattr("vimsam_segmenter.core.app.SegmenterApp", DummyApp)

    exit_code = main([
        "--input",
        "input.raw",
        "--out",
        str(output_path),
        "--workflow",
        "raw_timeseries_logits",
        "--fps",
        "12.5",
        "--timestamp-format",
        "",
        "--time-seconds",
        "0,0.5",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(output_path) in captured.out


def test_cli_passes_save_combined_video(monkeypatch, capsys, tmp_path):
    output_path = tmp_path / "result.png"

    class DummyApp:
        def run(self, config):
            assert config.workflow == "raw_timeseries_logits"
            assert config.save_combined_video is True
            return SegmentationResult(success=True, outputs=(output_path,))

    monkeypatch.setattr("vimsam_segmenter.core.app.SegmenterApp", DummyApp)

    exit_code = main([
        "--input",
        "input.raw",
        "--out",
        str(output_path),
        "--workflow",
        "raw_timeseries_logits",
        "--save-combined-video",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(output_path) in captured.out


def test_main_prints_result_details_and_returns_zero(monkeypatch, capsys, tmp_path):
    output_path = tmp_path / "result.png"
    stats_path = tmp_path / "stats.csv"

    class DummyApp:
        def run(self, config):
            assert config.workflow == "single"
            return SegmentationResult(
                success=True,
                message="completed",
                outputs=(output_path,),
                stats_path=stats_path,
            )

    monkeypatch.setattr("vimsam_segmenter.core.app.SegmenterApp", DummyApp)

    exit_code = main(["--input", "input.tif", "--out", str(output_path), "--workflow", "single"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "completed" in captured.out
    assert str(output_path) in captured.out
    assert str(stats_path) in captured.out


def test_main_reports_failed_result_to_stderr(monkeypatch, capsys, tmp_path):
    output_path = tmp_path / "result.png"

    class DummyApp:
        def run(self, config):
            return SegmentationResult(success=False, message="failed", outputs=(output_path,))

    monkeypatch.setattr("vimsam_segmenter.core.app.SegmenterApp", DummyApp)

    exit_code = main(["--input", "input.tif", "--out", str(output_path), "--workflow", "single"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "failed" in captured.err


def test_main_handles_segmenter_errors_without_exiting(monkeypatch, capsys, tmp_path):
    output_path = tmp_path / "result.png"

    class DummyApp:
        def run(self, config):
            raise SegmenterError("boom")

    monkeypatch.setattr("vimsam_segmenter.core.app.SegmenterApp", DummyApp)

    exit_code = main(["--input", "input.tif", "--out", str(output_path), "--workflow", "single"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Error: boom" in captured.err
