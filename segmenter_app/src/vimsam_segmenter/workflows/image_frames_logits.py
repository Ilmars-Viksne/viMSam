from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..core.config import SegmentationResult, WorkflowConfig
from ..io.local import list_files, load_image, output_dir_for, save_image, save_records
from ..io.series_outputs import (
    combined_frame_path,
    ensure_series_output_dirs,
    mask_frame_path,
)
from ..io.video_outputs import save_combined_video
from ..processing.preprocess import PreProcessor
from ..tracking.logit_propagation import LogitPropagationTracker
from ..utils.geometry import get_box_from_mask, get_centroid, get_pole_of_inaccessibility
from ..utils.logging import setup_logger
from ..utils.prompts import build_prompt_overlay
from ..utils.standard_stats import build_standard_stats_record
from ..utils.time_resolver import resolve_time_seconds
from ..utils.visualization import create_visualization

from .base import BaseWorkflow


logger = setup_logger("ImageFrameLogitsWorkflow")


class ImageFrameLogitsWorkflow(BaseWorkflow):
    IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")

    def run(self, config: WorkflowConfig) -> SegmentationResult:
        output_dir = output_dir_for(config.output_path)

        frame_files = list_files(config.input_path, self.IMAGE_PATTERNS)
        if not frame_files:
            return SegmentationResult(
                success=False,
                count=0,
                message=f"No image frames found in {config.input_path}",
            )

        predictor = self.model_service.get_predictor()
        preprocessor = PreProcessor(method=config.preprocessing_method)

        tracker = LogitPropagationTracker(
            fallback=config.tracking_method,
            reset_on_empty_mask=True,
        )

        records: list[dict[str, object]] = []
        outputs: list[Path] = []
        combined_video_frames: list[np.ndarray] = []
        need_combined_frame = config.save_combined or config.save_combined_video

        prompt_points = None
        prompt_box = None

        if config.prompts is not None:
            prompt_points = config.prompts.points
            prompt_box = config.prompts.box

        if not prompt_points and prompt_box is None:
            return SegmentationResult(
                success=False,
                count=0,
                message=(
                    "The first frame of image_frames_logits requires "
                    "at least one point prompt or a box prompt."
                ),
            )

        ensure_series_output_dirs(output_dir, save_combined=config.save_combined)
        times = resolve_time_seconds(
            source_paths=frame_files,
            fps=config.fps,
            timestamp_format=config.timestamp_format,
            user_time_seconds=config.time_seconds,
        )

        for frame_index, frame_path in enumerate(tqdm(frame_files, desc="Logit propagation")):
            image = load_image(frame_path)
            processed = preprocessor.run(image)

            predictor.set_image(self.sam_image(processed))

            if frame_index == 0:
                result = tracker.initialize_from_prompt(
                    predictor=predictor,
                    points=prompt_points,
                    box=prompt_box,
                )
            else:
                result = tracker.propagate(predictor=predictor)

            mask = result.mask

            mask_output_path = mask_frame_path(
                output_dir,
                frame_index=frame_index,
                mode=config.frame_name_mode,
                source_path=frame_path,
                prefix=config.frame_name_prefix,
            )

            combined_output_path = combined_frame_path(
                output_dir,
                frame_index=frame_index,
                mode=config.frame_name_mode,
                source_path=frame_path,
                prefix=config.frame_name_prefix,
            )            

            mask_image = create_visualization(
                processed,
                mask,
                prompts=None,
                save_combined=False,
                show_prompts=False,
            )
            save_image(mask_output_path, mask_image)
            outputs.append(mask_output_path)

            if need_combined_frame:
                current_prompt_overlay = {}
                if frame_index == 0:
                    current_prompt_overlay = build_prompt_overlay(
                        points=prompt_points,
                        box=prompt_box,
                    )
                elif result.used_fallback_prompt:
                    point = None
                    box = None
                    if config.tracking_method == "centroid":
                        point = get_centroid(mask)
                    elif config.tracking_method == "pole":
                        point = get_pole_of_inaccessibility(mask)
                    elif config.tracking_method == "box":
                        box = get_box_from_mask(mask)
                    current_prompt_overlay = build_prompt_overlay(
                        points=(point,) if point is not None else None,
                        box=box,
                    )

                combined = create_visualization(
                    processed,
                    mask,
                    prompts=current_prompt_overlay if config.show_prompts else None,
                    save_combined=True,
                    show_prompts=config.show_prompts,
                )
                if config.save_combined:
                    save_image(combined_output_path, combined)
                    outputs.append(combined_output_path)
                if config.save_combined_video:
                    combined_video_frames.append(combined)

            records.append(
                build_standard_stats_record(
                    source_path=frame_path,
                    time_seconds=times[frame_index],
                    frame_id=frame_index,
                    mask=mask,
                    mask_label=1,
                    iou_score=result.score,
                    has_combined=config.save_combined,
                    mask_output_name=mask_output_path.name,
                    combined_output_name=(
                        combined_output_path.name
                        if config.save_combined
                        else None
                    ),
                )
            )

        stats_path = save_records(
            output_dir / "stats",
            records,
            config.export_format,
        )

        if stats_path is not None:
            outputs.append(stats_path)

        if config.save_combined_video:
            outputs.append(save_combined_video(output_dir, combined_video_frames, fps=config.fps))

        return SegmentationResult(
            success=True,
            count=len(frame_files),
            message=(
                f"Processed {len(frame_files)} image frames using "
                "logits-based mask-input propagation."
            ),
            outputs=tuple(outputs),
            stats_path=stats_path,
            metadata={
                "workflow": "image_frames_logits",
                "tracking_method": config.tracking_method,
                "used_logits_propagation": True,
            },
        )

    def sam_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)

        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            return np.repeat(arr, 3, axis=-1)

        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr

        if arr.ndim == 3 and arr.shape[-1] == 4:
            return arr[..., :3]

        raise ValueError(f"Unsupported image shape for SAM input: {arr.shape}")


