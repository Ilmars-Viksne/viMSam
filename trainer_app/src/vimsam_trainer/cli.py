from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .core.app import TrainerApp
from .core.config import TrainingConfig
from .core.errors import TrainerError


def parse_patch_shape(value: str) -> tuple[int, int]:
    """
    Parse a patch shape in HEIGHT,WIDTH or HEIGHTxWIDTH format.

    Examples:
        512,512
        512x512
        256,512
    """
    parts = [
        part.strip()
        for part in value.lower().replace("x", ",").split(",")
    ]

    if len(parts) != 2 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "Expected HEIGHT,WIDTH, for example 512,512"
        )

    try:
        height, width = (
            int(part)
            for part in parts
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Patch shape values must be integers"
        ) from exc

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            "Patch shape values must be positive"
        )

    return height, width


def build_parser() -> argparse.ArgumentParser:
    """Build the viMSam Trainer command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="vimsam-trainer",
        description=(
            "Fine-tune micro-SAM models using microscopy images "
            "or raw uint16 frames."
        ),
    )

    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Directory containing image or raw-frame files.",
    )
    parser.add_argument(
        "--masks",
        required=True,
        type=Path,
        help="Directory containing the corresponding mask files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for checkpoints and training metadata.",
    )

    parser.add_argument(
        "--workflow",
        choices=[
            "raw_frames",
            "image_frames",
        ],
        default="raw_frames",
        help="Input-data workflow. Default: raw_frames.",
    )
    parser.add_argument(
        "--model",
        "--model-type",
        dest="model_type",
        default="vit_b",
        help="micro-SAM model type. Default: vit_b.",
    )
    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="Training device. Default: auto.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint from which training will resume.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs. Default: 50.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size. Default: 2.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Optimizer learning rate. Default: 1e-5.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting. Default: 42.",
    )

    parser.add_argument(
        "--raw-width",
        type=int,
        default=1024,
        help="Width of uint16 raw frames. Default: 1024.",
    )
    parser.add_argument(
        "--raw-height",
        type=int,
        default=1024,
        help="Height of uint16 raw frames. Default: 1024.",
    )
    parser.add_argument(
        "--preprocessing-method",
        choices=[
            "fixed_16bit",
            "minmax",
            "percentile",
            "none",
        ],
        default="fixed_16bit",
        help="Image preprocessing method. Default: fixed_16bit.",
    )
    parser.add_argument(
        "--patch-shape",
        type=parse_patch_shape,
        default=(512, 512),
        metavar="HEIGHT,WIDTH",
        help="Training patch shape. Default: 512,512.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data-loader workers. Default: 0.",
    )
    parser.add_argument(
        "--n-objects-per-batch",
        type=int,
        default=25,
        help=(
            "Maximum number of objects used for prompt-based "
            "training in each batch. Default: 25."
        ),
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=25,
        help=(
            "Minimum object size in pixels for an instance to be "
            "considered valid. Default: 25."
        ),
    )
    parser.add_argument(
        "--min-instances-per-patch",
        type=int,
        default=1,
        help=(
            "Minimum number of valid foreground instances required "
            "in a sampled patch. Default: 1."
        ),
    )
    parser.add_argument(
        "--without-segmentation-decoder",
        action="store_true",
        help=(
            "Disable training of the additional instance "
            "segmentation decoder."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the viMSam Trainer command-line application."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = TrainingConfig(
            workflow=args.workflow,
            images_path=args.images,
            masks_path=args.masks,
            output_path=args.out,
            model_type=args.model_type,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_fraction=args.val_fraction,
            seed=args.seed,
            raw_width=args.raw_width,
            raw_height=args.raw_height,
            preprocessing_method=args.preprocessing_method,
            patch_shape=args.patch_shape,
            num_workers=args.num_workers,
            n_objects_per_batch=args.n_objects_per_batch,
            min_size=args.min_size,
            min_instances_per_patch=(
                args.min_instances_per_patch
            ),
            with_segmentation_decoder=(
                not args.without_segmentation_decoder
            ),
        )

        result = TrainerApp().run(config)

    except TrainerError as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 2

    if not result.success:
        print(
            result.message or "Training failed",
            file=sys.stderr,
        )
        return 1

    if result.message:
        print(result.message)

    if result.output_dir is not None:
        print(
            f"Output directory: {result.output_dir}"
        )

    if result.checkpoint_path is not None:
        print(
            f"Checkpoint: {result.checkpoint_path}"
        )

    if result.summary_path is not None:
        print(
            f"Summary: {result.summary_path}"
        )

    print(
        f"Train samples: {result.train_count}"
    )
    print(
        f"Validation samples: {result.val_count}"
    )

    return 0