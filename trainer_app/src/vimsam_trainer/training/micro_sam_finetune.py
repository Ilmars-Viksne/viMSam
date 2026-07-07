from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from ..core.config import TrainingConfig
from ..core.errors import DependencyMissingError, OutputWriteError
from ..core.result import TrainingResult
from ..data.pairs import ImageMaskPair
from ..data.raw_io import read_u16_raw
from ..preprocessing.preprocess import PreProcessor
from ..utils.logging import setup_logger
from ..utils.paths import ensure_output_dir

logger = setup_logger(__name__)


class MicroSamFineTuner:
    """Adapter around micro_sam.training for viMSam fine-tuning workflows."""

    def run(
        self,
        *,
        config: TrainingConfig,
        train_pairs: list[ImageMaskPair],
        val_pairs: list[ImageMaskPair],
    ) -> TrainingResult:
        output_dir = ensure_output_dir(Path(config.output_path))

        try:
            from micro_sam.training import default_sam_loader, train_sam
        except ImportError as exc:
            raise DependencyMissingError(
                "micro_sam with training dependencies is required. "
                "Install trainer_app with the ml extra or use the Colab/mamba environment."
            ) from exc

        try:
            with tempfile.TemporaryDirectory(prefix="vimsam_trainer_") as tmp:
                tmp_dir = Path(tmp)
                prepared_train_images, prepared_train_masks = self._prepare_dataset(
                    config=config,
                    pairs=train_pairs,
                    target_dir=tmp_dir / "train",
                )
                prepared_val_images, prepared_val_masks = self._prepare_dataset(
                    config=config,
                    pairs=val_pairs,
                    target_dir=tmp_dir / "val",
                )

                train_loader = default_sam_loader(
                    raw_paths=prepared_train_images,
                    raw_key=None,
                    label_paths=prepared_train_masks,
                    label_key=None,
                    patch_shape=config.patch_shape,
                    with_segmentation_decoder=config.with_segmentation_decoder,
                    min_size=config.min_size,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                )
                val_loader = default_sam_loader(
                    raw_paths=prepared_val_images,
                    raw_key=None,
                    label_paths=prepared_val_masks,
                    label_key=None,
                    patch_shape=config.patch_shape,
                    with_segmentation_decoder=config.with_segmentation_decoder,
                    min_size=config.min_size,
                    batch_size=1,
                    shuffle=False,
                    num_workers=config.num_workers,
                    is_train=False,
                )

                train_sam(
                    name=output_dir.name,
                    model_type=config.model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=config.epochs,
                    n_objects_per_batch=config.n_objects_per_batch,
                    checkpoint_path=(
                        str(config.checkpoint_path)
                        if config.checkpoint_path is not None
                        else None
                    ),
                    with_segmentation_decoder=config.with_segmentation_decoder,
                    device=None if config.device == "auto" else config.device,
                    lr=config.learning_rate,
                    save_root=str(output_dir),
                )

                summary_path = self._write_summary(
                    output_dir=output_dir,
                    config=config,
                    train_count=len(train_pairs),
                    val_count=len(val_pairs),
                )
                checkpoint_path = self._find_best_checkpoint(output_dir)

                return TrainingResult(
                    success=True,
                    count=len(train_pairs) + len(val_pairs),
                    message="Fine-tuning completed successfully.",
                    output_dir=output_dir,
                    checkpoint_path=checkpoint_path,
                    summary_path=summary_path,
                    train_count=len(train_pairs),
                    val_count=len(val_pairs),
                    metadata={
                        "workflow": config.workflow,
                        "model_type": config.model_type,
                        "patch_shape": config.patch_shape,
                    },
                )
        except DependencyMissingError:
            raise
        except Exception as exc:
            raise OutputWriteError(f"Fine-tuning failed: {exc}") from exc

    def _prepare_dataset(
        self,
        *,
        config: TrainingConfig,
        pairs: list[ImageMaskPair],
        target_dir: Path,
    ) -> tuple[list[str], list[str]]:
        import imageio.v3 as imageio

        images_dir = target_dir / "images"
        masks_dir = target_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        preprocessor = PreProcessor(config.preprocessing_method)
        prepared_images: list[str] = []
        prepared_masks: list[str] = []

        for index, pair in enumerate(pairs):
            stem = f"sample_{index:05d}"
            image_out = images_dir / f"{stem}.tif"
            mask_out = masks_dir / f"{stem}.tif"

            if config.workflow == "raw_frames":
                image = read_u16_raw(
                    pair.image_path,
                    width=config.raw_width,
                    height=config.raw_height,
                )
            else:
                image = imageio.imread(pair.image_path)

            image = preprocessor.run(image)
            mask = self._normalize_mask(imageio.imread(pair.mask_path))

            imageio.imwrite(image_out, np.ascontiguousarray(image))
            imageio.imwrite(mask_out, np.ascontiguousarray(mask))
            prepared_images.append(str(image_out))
            prepared_masks.append(str(mask_out))

        return prepared_images, prepared_masks

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask)
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
        if mask.dtype == bool:
            return mask.astype(np.uint32)
        if np.issubdtype(mask.dtype, np.integer):
            return mask.astype(np.uint32)
        return (mask > 0).astype(np.uint32)

    @staticmethod
    def _write_summary(
        *,
        output_dir: Path,
        config: TrainingConfig,
        train_count: int,
        val_count: int,
    ) -> Path:
        summary = {
            "workflow": config.workflow,
            "model_type": config.model_type,
            "device": config.device,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "val_fraction": config.val_fraction,
            "train_count": train_count,
            "val_count": val_count,
            "raw_width": config.raw_width,
            "raw_height": config.raw_height,
            "preprocessing_method": config.preprocessing_method,
            "patch_shape": list(config.patch_shape),
            "with_segmentation_decoder": config.with_segmentation_decoder,
            "min_size": config.min_size,
        }
        path = output_dir / "training_summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=4)
            handle.write("\n")
        return path

    @staticmethod
    def _find_best_checkpoint(output_dir: Path) -> Path | None:
        checkpoint_dir = output_dir / "checkpoints" / output_dir.name
        candidates = sorted(checkpoint_dir.glob("best.pt"))
        if candidates:
            return candidates[0]
        candidates = sorted(output_dir.rglob("*.pt"))
        if candidates:
            return candidates[0]
        return None
