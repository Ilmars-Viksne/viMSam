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

            if image.shape[:2] != mask.shape:
                raise ValueError(
                    f"Image/mask shape mismatch for {pair.image_path.name}: "
                    f"image shape is {image.shape[:2]}, "
                    f"mask shape is {mask.shape}."
                )

            imageio.imwrite(image_out, np.ascontiguousarray(image))
            imageio.imwrite(mask_out, np.ascontiguousarray(mask))
            prepared_images.append(str(image_out))
            prepared_masks.append(str(mask_out))

        return prepared_images, prepared_masks

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        """
        Convert a mask to a 2D uint32 instance-label image.

        Supported inputs:
        - (H, W): grayscale or label mask
        - (H, W, 1): single-channel mask
        - (H, W, 3): RGB mask
        - (H, W, 4): RGBA mask; alpha is ignored
        - (1, H, W), (3, H, W), (4, H, W): channel-first variants

        For RGB masks:
        - If all RGB channels are identical, the first channel is used.
        - Otherwise, each unique RGB color is converted to an instance ID.
        - Black is treated as background label 0.
        """
        mask = np.asarray(mask)

        # Remove dimensions whose size is one, for example (H, W, 1).
        mask = np.squeeze(mask)

        # Convert channel-first masks, such as (3, H, W), to channel-last.
        if (
            mask.ndim == 3
            and mask.shape[0] in (1, 3, 4)
            and mask.shape[-1] not in (1, 3, 4)
        ):
            mask = np.moveaxis(mask, 0, -1)

        if mask.ndim == 3:
            if mask.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    "Expected a 2D mask or an RGB/RGBA mask, "
                    f"got shape {mask.shape}"
                )

            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                # Ignore alpha if this is an RGBA mask.
                rgb = mask[..., :3]

                # Many grayscale TIFF masks are stored as RGB with the same
                # value repeated in all three channels.
                if (
                    np.array_equal(rgb[..., 0], rgb[..., 1])
                    and np.array_equal(rgb[..., 0], rgb[..., 2])
                ):
                    mask = rgb[..., 0]
                else:
                    mask = MicroSamFineTuner._rgb_mask_to_instances(rgb)

        if mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D mask after normalization, got shape {mask.shape}"
            )

        if mask.dtype == bool:
            return np.ascontiguousarray(mask.astype(np.uint32))

        if np.issubdtype(mask.dtype, np.integer):
            if np.issubdtype(mask.dtype, np.signedinteger) and np.any(mask < 0):
                raise ValueError("Mask labels must not contain negative values")

            mask = mask.astype(np.uint32, copy=False)

            # Normalize ordinary binary masks, such as {0, 255}, to {0, 1}.
            unique_values = np.unique(mask)
            if unique_values.size <= 2 and 0 in unique_values:
                mask = (mask != 0).astype(np.uint32)

            return np.ascontiguousarray(mask)

        if np.issubdtype(mask.dtype, np.floating):
            if not np.all(np.isfinite(mask)):
                raise ValueError("Mask contains NaN or infinite values")

            # Preserve integer-valued floating-point instance masks.
            if np.all(mask >= 0) and np.all(mask == np.floor(mask)):
                return np.ascontiguousarray(mask.astype(np.uint32))

            # Otherwise interpret the floating-point mask as binary.
            return np.ascontiguousarray((mask > 0).astype(np.uint32))

        raise ValueError(f"Unsupported mask dtype: {mask.dtype}")

    @staticmethod
    def _rgb_mask_to_instances(rgb: np.ndarray) -> np.ndarray:
        """
        Convert a color-coded RGB mask to a 2D instance-label mask.

        Black (0, 0, 0) is background. Every other unique color receives
        a positive integer label.
        """
        rgb = np.asarray(rgb)

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected an RGB mask, got shape {rgb.shape}")

        height, width, _ = rgb.shape
        pixels = np.ascontiguousarray(rgb).reshape(-1, 3)

        colors, inverse = np.unique(pixels, axis=0, return_inverse=True)

        labels_for_colors = np.zeros(len(colors), dtype=np.uint32)
        next_label = 1

        for color_index, color in enumerate(colors):
            if np.all(color == 0):
                # Black remains background.
                continue

            labels_for_colors[color_index] = next_label
            next_label += 1

        instance_mask = labels_for_colors[inverse].reshape(height, width)
        return np.ascontiguousarray(instance_mask)

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
