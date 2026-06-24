from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .core.config import ModelConfig, PromptConfig, WorkflowConfig
from .core.errors import SegmenterError


def configure_ffmpeg() -> None:
    system_ffmpeg = "/usr/bin/ffmpeg"
    if os.path.exists(system_ffmpeg):
        os.environ["IMAGEIO_FFMPEG_EXE"] = system_ffmpeg


def parse_points(points_str: str | None) -> tuple[tuple[int, int], ...] | None:
    if points_str is None or not points_str.strip():
        return None
    points: list[tuple[int, int]] = []
    for group in points_str.strip().split():
        parts = group.split(",")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise argparse.ArgumentTypeError(f"Invalid point '{group}'. Expected x,y, for example 500,480.")
        try:
            points.append((int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid point '{group}'. Coordinates must be integers.") from exc
    return tuple(points)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vimsam-segmenter")
    parser.add_argument("--input", required=True, help="Input file or directory for time-series workflows")
    parser.add_argument("--out", required=True, help="Output file or directory")
    parser.add_argument("--workflow", choices=["single", "video", "raw_single", "raw_timeseries"], default="single")
    parser.add_argument("--points", type=parse_points, default=None)
    parser.add_argument("--tracking_method", "--tracking-method", dest="tracking_method", choices=["box", "centroid", "pole"], default="box")
    parser.add_argument("--show_prompts", "--show-prompts", dest="show_prompts", action="store_true", help="Draw prompts on combined output")
    parser.add_argument("--save_combined", "--save-combined", dest="save_combined", action="store_true", help="Save combined original/mask/overlay output")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--model", "--model-type", dest="model_type", default="vit_b")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--raw-width", type=int, default=1024)
    parser.add_argument("--raw-height", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_ffmpeg()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from .core.app import SegmenterApp

        prompts = PromptConfig(points=args.points) if args.points else None
        config = WorkflowConfig(
            workflow=args.workflow,
            input_path=args.input,
            output_path=args.out,
            model=ModelConfig(
                model_type=args.model_type,
                device=args.device,
                checkpoint_path=args.checkpoint_path,
            ),
            prompts=prompts,
            show_prompts=args.show_prompts,
            save_combined=args.save_combined,
            tracking_method=args.tracking_method,
            export_format=args.format,
            raw_width=args.raw_width,
            raw_height=args.raw_height,
        )
        result = SegmenterApp().run(config)
    except SegmenterError as exc:
        parser.exit(2, f"error: {exc}\n")

    if result.message:
        print(result.message)
    return 0 if result.success else 1
