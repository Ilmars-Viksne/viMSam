from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

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
    """Adapter around micro_sam.training for viMSam fine-tuning."""

    def run(
        self,
        *,
        config: TrainingConfig,
        train_pairs: list[ImageMaskPair],
        val_pairs: list[ImageMaskPair],
    ) -> TrainingResult:
        """
        Prepare the training data and run micro-SAM fine-tuning.

        Args:
            config:
                Fine-tuning configuration.
            train_pairs:
                Image/mask pairs used for training.
            val_pairs:
                Image/mask pairs used for validation.

        Returns:
            The fine-tuning result.
        """
        output_dir = ensure_output_dir(Path(config.output_path))

        try:
            from micro_sam.training import (
                default_sam_dataset,
                train_sam,
            )
            from torch.utils.data import DataLoader
            from torch_em.data.sampler import MinInstanceSampler
        except ImportError as exc:
            raise DependencyMissingError(
                "micro_sam with training dependencies is required. "
                "Install trainer_app with the ml extra or use the "
                "Colab/mamba environment."
            ) from exc

        try:
            with tempfile.TemporaryDirectory(
                prefix="vimsam_trainer_"
            ) as tmp:
                tmp_dir = Path(tmp)

                prepared_train_images, prepared_train_masks = (
                    self._prepare_dataset(
                        config=config,
                        pairs=train_pairs,
                        target_dir=tmp_dir / "train",
                    )
                )

                prepared_val_images, prepared_val_masks = (
                    self._prepare_dataset(
                        config=config,
                        pairs=val_pairs,
                        target_dir=tmp_dir / "val",
                    )
                )

                train_sampler = MinInstanceSampler(
                    min_num_instances=config.min_instances_per_patch,
                    min_size=config.min_size,
                )

                val_sampler = MinInstanceSampler(
                    min_num_instances=config.min_instances_per_patch,
                    min_size=config.min_size,
                )

                train_dataset = default_sam_dataset(
                    raw_paths=prepared_train_images,
                    raw_key=None,
                    label_paths=prepared_train_masks,
                    label_key=None,
                    patch_shape=config.patch_shape,
                    with_segmentation_decoder=(
                        config.with_segmentation_decoder
                    ),
                    sampler=train_sampler,
                    is_train=True,
                )

                val_dataset = default_sam_dataset(
                    raw_paths=prepared_val_images,
                    raw_key=None,
                    label_paths=prepared_val_masks,
                    label_key=None,
                    patch_shape=config.patch_shape,
                    with_segmentation_decoder=(
                        config.with_segmentation_decoder
                    ),
                    sampler=val_sampler,
                    is_train=False,
                )

                use_pin_memory = config.device == "cuda"

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    pin_memory=use_pin_memory,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=use_pin_memory,
                )

                # Force one batch to be sampled before model
                # initialization. This produces an early, localized
                # error if patch sampling or label transformation fails.
                self._validate_loader_batch(
                    loader=train_loader,
                    loader_name="training",
                    with_segmentation_decoder=(
                        config.with_segmentation_decoder
                    ),
                )

                self._validate_loader_batch(
                    loader=val_loader,
                    loader_name="validation",
                    with_segmentation_decoder=(
                        config.with_segmentation_decoder
                    ),
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
                    with_segmentation_decoder=(
                        config.with_segmentation_decoder
                    ),
                    device=(
                        None
                        if config.device == "auto"
                        else config.device
                    ),
                    lr=config.learning_rate,
                    save_root=str(output_dir),
                )

                summary_path = self._write_summary(
                    output_dir=output_dir,
                    config=config,
                    train_count=len(train_pairs),
                    val_count=len(val_pairs),
                )

                checkpoint_path = self._find_best_checkpoint(
                    output_dir
                )

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
                        "min_size": config.min_size,
                        "min_instances_per_patch": (
                            config.min_instances_per_patch
                        ),
                    },
                )

        except Exception as exc:
            raise OutputWriteError(
                f"Fine-tuning failed: {exc}"
            ) from exc

    def _prepare_dataset(
        self,
        *,
        config: TrainingConfig,
        pairs: list[ImageMaskPair],
        target_dir: Path,
    ) -> tuple[list[str], list[str]]:
        """
        Convert source images and masks into temporary TIFF files.

        The resulting masks are 2D uint32 instance-label images
        with background label zero.
        """
        import imageio.v3 as imageio

        images_dir = target_dir / "images"
        masks_dir = target_dir / "masks"

        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        preprocessor = PreProcessor(
            config.preprocessing_method
        )

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

            mask = self._normalize_mask(
                imageio.imread(pair.mask_path),
                min_size=config.min_size,
            )

            image_spatial_shape = self._get_image_spatial_shape(
                image,
                path=pair.image_path,
            )

            if image_spatial_shape != mask.shape:
                raise ValueError(
                    "Image/mask shape mismatch for "
                    f"{pair.image_path.name}: "
                    f"image={image_spatial_shape}, "
                    f"mask={mask.shape}"
                )

            if (
                config.patch_shape[0] > image_spatial_shape[0]
                or config.patch_shape[1] > image_spatial_shape[1]
            ):
                raise ValueError(
                    f"Patch shape {config.patch_shape} is larger "
                    f"than image {pair.image_path.name}, which has "
                    f"shape {image_spatial_shape}"
                )

            self._validate_prepared_mask(
                mask,
                path=pair.mask_path,
                min_size=config.min_size,
                min_instances=config.min_instances_per_patch,
            )

            imageio.imwrite(
                image_out,
                np.ascontiguousarray(image),
            )
            imageio.imwrite(
                mask_out,
                np.ascontiguousarray(mask),
            )

            prepared_images.append(str(image_out))
            prepared_masks.append(str(mask_out))

        return prepared_images, prepared_masks

    @staticmethod
    def _get_image_spatial_shape(
        image: np.ndarray,
        *,
        path: Path,
    ) -> tuple[int, int]:
        """Return the height and width of a supported image array."""
        image = np.asarray(image)

        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])

        if image.ndim == 3:
            # Channel-last image: (H, W, C).
            if image.shape[-1] in (1, 3, 4):
                return int(image.shape[0]), int(image.shape[1])

            # Channel-first image: (C, H, W).
            if image.shape[0] in (1, 3, 4):
                return int(image.shape[1]), int(image.shape[2])

        raise ValueError(
            f"Unsupported image shape for {path.name}: "
            f"{image.shape}. Expected (H, W), (H, W, C), "
            "or (C, H, W) with 1, 3, or 4 channels."
        )

    @staticmethod
    def _validate_prepared_mask(
        mask: np.ndarray,
        *,
        path: Path,
        min_size: int,
        min_instances: int,
    ) -> None:
        """Validate a normalized instance-label mask."""
        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                f"Prepared mask {path.name} is not 2D: "
                f"shape={mask.shape}"
            )

        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(
                f"Prepared mask {path.name} must have an "
                f"integer dtype, got {mask.dtype}"
            )

        if np.any(mask < 0):
            raise ValueError(
                f"Prepared mask {path.name} contains "
                "negative instance labels"
            )

        labels, counts = np.unique(
            mask,
            return_counts=True,
        )

        foreground = labels != 0
        foreground_labels = labels[foreground]
        foreground_sizes = counts[foreground]

        if foreground_labels.size == 0:
            raise ValueError(
                f"Mask {path.name} contains no foreground instances"
            )

        valid = foreground_sizes >= min_size
        valid_labels = foreground_labels[valid]
        valid_sizes = foreground_sizes[valid]

        if valid_labels.size == 0:
            raise ValueError(
                f"Mask {path.name} contains foreground objects, "
                f"but none has at least {min_size} pixels. "
                f"Object sizes: {foreground_sizes.tolist()}"
            )

        if valid_labels.size < min_instances:
            raise ValueError(
                f"Mask {path.name} contains only "
                f"{valid_labels.size} valid instances, but "
                f"{min_instances} instances are required per "
                "sampled patch."
            )

        logger.info(
            "Prepared mask %s: shape=%s, instances=%d, "
            "valid_instances=%d, foreground_pixels=%d, "
            "smallest=%d, largest=%d",
            path.name,
            mask.shape,
            len(foreground_labels),
            len(valid_labels),
            int(foreground_sizes.sum()),
            int(valid_sizes.min()),
            int(valid_sizes.max()),
        )

    @staticmethod
    def _normalize_mask(
        mask: np.ndarray,
        *,
        min_size: int = 1,
    ) -> np.ndarray:
        """
        Convert a mask into a 2D uint32 instance-label mask.

        Supported formats:

        - ``(H, W)`` binary masks
        - ``(H, W)`` integer instance-label masks
        - ``(H, W, 1)`` single-channel masks
        - ``(H, W, 3)`` RGB masks
        - ``(H, W, 4)`` RGBA masks
        - Channel-first variants such as ``(3, H, W)``

        Binary masks are split into connected components.
        Repeated-channel RGB masks are interpreted as grayscale.
        Color-coded RGB masks assign one instance ID to every
        connected component of every foreground color.
        """
        from skimage.measure import label

        if min_size <= 0:
            raise ValueError("min_size must be positive")

        mask = np.asarray(mask)
        mask = np.squeeze(mask)

        # Convert channel-first masks such as (3, H, W)
        # to channel-last format (H, W, 3).
        if (
            mask.ndim == 3
            and mask.shape[0] in (1, 3, 4)
            and mask.shape[-1] not in (1, 3, 4)
        ):
            mask = np.moveaxis(mask, 0, -1)

        if mask.ndim == 3:
            if mask.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    "Expected a 2D, RGB, or RGBA mask, "
                    f"got shape {mask.shape}"
                )

            if mask.shape[-1] == 1:
                foreground = mask[..., 0] != 0
                instances = label(
                    foreground,
                    connectivity=1,
                )
            else:
                # Ignore the alpha channel in RGBA masks.
                rgb = np.ascontiguousarray(mask[..., :3])

                channels_identical = (
                    np.array_equal(rgb[..., 0], rgb[..., 1])
                    and np.array_equal(rgb[..., 0], rgb[..., 2])
                )

                if channels_identical:
                    # Grayscale mask saved with repeated RGB channels.
                    foreground = rgb[..., 0] != 0
                    instances = label(
                        foreground,
                        connectivity=1,
                    )
                else:
                    instances = (
                        MicroSamFineTuner._rgb_mask_to_instances(rgb)
                    )

        elif mask.ndim == 2:
            if np.issubdtype(mask.dtype, np.floating):
                if not np.all(np.isfinite(mask)):
                    raise ValueError(
                        "Mask contains NaN or infinite values"
                    )

                if np.any(mask < 0):
                    raise ValueError(
                        "Mask labels must not contain negative values"
                    )

            if np.issubdtype(mask.dtype, np.signedinteger):
                if np.any(mask < 0):
                    raise ValueError(
                        "Mask labels must not contain negative values"
                    )

            if mask.dtype == bool:
                instances = label(
                    mask,
                    connectivity=1,
                )
            else:
                unique_values = np.unique(mask)

                if unique_values.size == 1:
                    if unique_values[0] == 0:
                        instances = np.zeros(
                            mask.shape,
                            dtype=np.uint32,
                        )
                    else:
                        instances = label(
                            mask != 0,
                            connectivity=1,
                        )

                elif (
                    unique_values.size == 2
                    and 0 in unique_values
                ):
                    # Binary masks such as {0, 1} or {0, 255}.
                    instances = label(
                        mask != 0,
                        connectivity=1,
                    )

                else:
                    # Preserve existing instance IDs.
                    if not np.issubdtype(
                        mask.dtype,
                        np.integer,
                    ):
                        integer_valued = np.all(
                            mask == np.floor(mask)
                        )

                        if not integer_valued:
                            raise ValueError(
                                "Non-binary floating-point "
                                "mask labels must be integer-valued"
                            )

                    instances = mask.astype(
                        np.int64,
                        copy=False,
                    )

        else:
            raise ValueError(
                "Expected a 2D, RGB, or RGBA mask, "
                f"got shape {mask.shape}"
            )

        return MicroSamFineTuner._remove_small_instances(
            instances,
            min_size=min_size,
        )

    @staticmethod
    def _remove_small_instances(
        instances: np.ndarray,
        *,
        min_size: int,
    ) -> np.ndarray:
        """
        Remove instances smaller than ``min_size`` pixels.

        Remaining labels are relabeled consecutively, with zero
        preserved as the background label.
        """
        from skimage.segmentation import relabel_sequential

        if min_size <= 0:
            raise ValueError("min_size must be positive")

        instances = np.asarray(instances)

        if instances.ndim != 2:
            raise ValueError(
                "Expected a 2D instance mask, "
                f"got shape {instances.shape}"
            )

        if np.any(instances < 0):
            raise ValueError(
                "Instance labels must not be negative"
            )

        instances = instances.astype(
            np.uint32,
            copy=True,
        )

        labels, counts = np.unique(
            instances,
            return_counts=True,
        )

        for instance_id, pixel_count in zip(labels, counts):
            if instance_id == 0:
                continue

            if pixel_count < min_size:
                instances[instances == instance_id] = 0

        instances, _, _ = relabel_sequential(instances)

        return np.ascontiguousarray(
            instances,
            dtype=np.uint32,
        )

    @staticmethod
    def _rgb_mask_to_instances(
        rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a color-coded RGB mask to instance labels.

        Black is treated as background when black is present.
        Otherwise, the most frequent color is treated as the
        background.

        Every connected component of every remaining color
        receives a separate positive instance ID.
        """
        from skimage.measure import label

        rgb = np.asarray(rgb)

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected an RGB mask, got shape {rgb.shape}"
            )

        height, width, _ = rgb.shape

        pixels = np.ascontiguousarray(rgb).reshape(-1, 3)

        colors, inverse, counts = np.unique(
            pixels,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        black_indices = np.flatnonzero(
            np.all(colors == 0, axis=1)
        )

        if black_indices.size:
            background_index = int(black_indices[0])
        else:
            # If there is no black, assume the most frequent
            # color is the background.
            background_index = int(np.argmax(counts))

            logger.warning(
                "RGB mask contains no black background. "
                "Using the most frequent color %s as background.",
                colors[background_index].tolist(),
            )

        color_indices = inverse.reshape(height, width)

        result = np.zeros(
            (height, width),
            dtype=np.uint32,
        )

        next_instance_id = 1

        for color_index in range(len(colors)):
            if color_index == background_index:
                continue

            color_region = color_indices == color_index

            components = label(
                color_region,
                connectivity=1,
            )

            component_count = int(components.max())

            for component_id in range(
                1,
                component_count + 1,
            ):
                component = components == component_id

                if not np.any(component):
                    continue

                result[component] = next_instance_id
                next_instance_id += 1

        return np.ascontiguousarray(
            result,
            dtype=np.uint32,
        )

    @staticmethod
    def _validate_loader_batch(
        *,
        loader: Any,
        loader_name: str,
        with_segmentation_decoder: bool,
    ) -> None:
        """Sample and validate one loader batch before training."""
        try:
            raw, labels = next(iter(loader))
        except Exception as exc:
            raise ValueError(
                f"Could not sample a valid {loader_name} batch: "
                f"{exc}"
            ) from exc

        if raw.ndim != 4:
            raise ValueError(
                f"The {loader_name} raw batch must have shape "
                f"(B, C, H, W), got {tuple(raw.shape)}"
            )

        raw_channels = int(raw.shape[1])

        if raw_channels not in (1, 3):
            raise ValueError(
                f"The {loader_name} raw batch must contain "
                f"1 or 3 channels, got {raw_channels}"
            )

        raw_min = float(raw.min())
        raw_max = float(raw.max())

        if raw_min < 0 or raw_max > 255:
            raise ValueError(
                f"The {loader_name} raw batch must be in "
                f"the range [0, 255], got "
                f"[{raw_min}, {raw_max}]"
            )

        if raw_max < 1:
            raise ValueError(
                f"The {loader_name} raw batch contains "
                "no non-zero image intensity"
            )

        if labels.ndim != 4:
            raise ValueError(
                f"The {loader_name} label batch must have "
                f"shape (B, C, H, W), got "
                f"{tuple(labels.shape)}"
            )

        expected_label_channels = (
            4 if with_segmentation_decoder else 1
        )

        actual_label_channels = int(labels.shape[1])

        if actual_label_channels != expected_label_channels:
            raise ValueError(
                f"The {loader_name} label batch has "
                f"{actual_label_channels} channels; expected "
                f"{expected_label_channels}"
            )

        instance_channel = labels[:, 0]

        for batch_index, instance_mask in enumerate(
            instance_channel
        ):
            unique_labels = instance_mask.unique()

            if unique_labels.numel() < 2:
                raise ValueError(
                    f"The {loader_name} sample at batch index "
                    f"{batch_index} contains no foreground "
                    "instance"
                )

            if bool((unique_labels < 0).any()):
                raise ValueError(
                    f"The {loader_name} sample at batch index "
                    f"{batch_index} contains negative labels"
                )

        logger.info(
            "%s batch validated: raw=%s, labels=%s, "
            "raw_range=[%.1f, %.1f]",
            loader_name.capitalize(),
            tuple(raw.shape),
            tuple(labels.shape),
            raw_min,
            raw_max,
        )

    @staticmethod
    def _write_summary(
        *,
        output_dir: Path,
        config: TrainingConfig,
        train_count: int,
        val_count: int,
    ) -> Path:
        """Write the training configuration summary."""
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
            "preprocessing_method": (
                config.preprocessing_method
            ),
            "patch_shape": list(config.patch_shape),
            "with_segmentation_decoder": (
                config.with_segmentation_decoder
            ),
            "min_size": config.min_size,
            "min_instances_per_patch": (
                config.min_instances_per_patch
            ),
            "n_objects_per_batch": (
                config.n_objects_per_batch
            ),
            "num_workers": config.num_workers,
        }

        path = output_dir / "training_summary.json"

        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                indent=4,
            )
            handle.write("\n")

        return path

    @staticmethod
    def _find_best_checkpoint(
        output_dir: Path,
    ) -> Path | None:
        """Find the best checkpoint written by micro-SAM."""
        checkpoint_dir = (
            output_dir
            / "checkpoints"
            / output_dir.name
        )

        candidates = sorted(
            checkpoint_dir.glob("best.pt")
        )

        if candidates:
            return candidates[0]

        candidates = sorted(
            output_dir.rglob("*.pt")
        )

        if candidates:
            return candidates[0]

        return None