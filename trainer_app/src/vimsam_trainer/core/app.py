from __future__ import annotations

from ..data.pairs import collect_image_mask_pairs
from ..data.splits import split_pairs
from ..utils.logging import setup_logger
from .config import TrainingConfig
from .result import TrainingResult

logger = setup_logger(__name__)


class TrainerApp:
    """Main application orchestrator for viMSam model fine-tuning."""

    def __init__(self, fine_tuner: object | None = None) -> None:
        self.fine_tuner = fine_tuner

    def run(self, config: TrainingConfig) -> TrainingResult:
        logger.info("Collecting image/mask pairs")
        pairs = collect_image_mask_pairs(
            images_path=config.images_path,
            masks_path=config.masks_path,
            workflow=config.workflow,
        )

        if not pairs:
            return TrainingResult(
                success=False,
                message="No image/mask pairs found.",
                output_dir=config.output_path,
            )

        logger.info("Splitting dataset")
        train_pairs, val_pairs = split_pairs(
            pairs,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )

        if not train_pairs or not val_pairs:
            return TrainingResult(
                success=False,
                message="Train/validation split produced an empty subset.",
                output_dir=config.output_path,
                train_count=len(train_pairs),
                val_count=len(val_pairs),
            )

        if self.fine_tuner is None:
            from ..training.micro_sam_finetune import MicroSamFineTuner

            fine_tuner = MicroSamFineTuner()
        else:
            fine_tuner = self.fine_tuner

        return fine_tuner.run(
            config=config,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
        )
