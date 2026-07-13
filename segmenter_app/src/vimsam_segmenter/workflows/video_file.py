from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from ..core.config import SegmentationResult, WorkflowConfig
from ..core.errors import InputValidationError
from ..io.local import (
    output_dir_for,
    read_metadata,
    save_image,
    save_records,
    stream_video,
)
from ..io.series_outputs import (
    combined_frame_path,
    ensure_series_output_dirs,
    mask_frame_path,
    video_source_stem,
)
from ..io.video_outputs import save_combined_video
from ..processing.preprocess import PreProcessor
from ..utils.geometry import (
    get_box_from_mask,
    get_centroid,
    get_pole_of_inaccessibility,
)
from ..utils.logging import setup_logger
from ..utils.prompts import build_prompt_overlay
from ..utils.standard_stats import build_standard_stats_record
from ..utils.visualization import create_visualization

from .base import BaseWorkflow, automatic_mask_generator


logger = setup_logger("VideoWorkflow")


class VideoFileWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        output_dir = output_dir_for(config.output_path)

        metadata = read_metadata(config.input_path)
        source_fps = metadata.get("fps", 5)
        video_fps = config.fps or float(source_fps or 5)

        predictor = self.model_service.get_predictor()
        preprocessor = PreProcessor(
            method=config.preprocessing_method,
        )

        records: list[dict[str, object]] = []
        outputs: list[Path] = []
        combined_video_frames: list[np.ndarray] = []

        current_logits = None
        current_mask = None
        frame_count = 0

        is_tracking = bool(
            config.prompts
            and config.prompts.points
        )

        points = (
            np.asarray(config.prompts.points)
            if is_tracking and config.prompts is not None
            else None
        )

        need_combined_frame = (
            config.save_combined
            or config.save_combined_video
        )

        ensure_series_output_dirs(
            output_dir,
            save_combined=config.save_combined,
        )

        logger.info(
            "Starting video processing. Tracking: %s. FPS: %s",
            is_tracking,
            video_fps,
        )

        def frame_generator() -> Iterator[
            np.ndarray | tuple[np.ndarray, np.ndarray]
        ]:
            nonlocal frame_count
            nonlocal current_logits
            nonlocal current_mask

            video_frames = enumerate(
                stream_video(config.input_path)
            )

            for frame_index, frame in tqdm(
                video_frames,
                desc="Processing Video",
            ):
                # Video frames do not have individual source filenames.
                # Construct a stable virtual source stem from the video
                # filename and the zero-based frame index.
                source_stem = video_source_stem(
                    config.input_path,
                    frame_index,
                )

                # Resolve both output paths before creating statistics.
                mask_output_path = mask_frame_path(
                    output_dir,
                    frame_index=frame_index,
                    mode=config.frame_name_mode,
                    source_stem=source_stem,
                    prefix=config.frame_name_prefix,
                )

                combined_output_path = combined_frame_path(
                    output_dir,
                    frame_index=frame_index,
                    mode=config.frame_name_mode,
                    source_stem=source_stem,
                    prefix=config.frame_name_prefix,
                )

                processed_frame = preprocessor.run(frame)

                predictor.set_image(
                    self.sam_image(processed_frame)
                )

                current_prompt_overlay = {}
                iou_score = 0.0

                if is_tracking:
                    if frame_index == 0:
                        masks, ious, logits = predictor.predict(
                            point_coords=points,
                            point_labels=np.ones(
                                len(points),
                                dtype=int,
                            ),
                            multimask_output=False,
                        )

                        current_mask = masks[0]
                        current_logits = logits
                        iou_score = float(ious[0])

                        current_prompt_overlay = (
                            build_prompt_overlay(
                                points=(
                                    tuple(map(tuple, points))
                                    if points is not None
                                    else None
                                ),
                                box=None,
                            )
                        )

                    elif (
                        current_mask is not None
                        and np.any(current_mask)
                    ):
                        next_point, next_box = self._next_prompt(
                            current_mask,
                            config.tracking_method,
                        )

                        current_prompt_overlay = (
                            build_prompt_overlay(
                                points=(
                                    tuple(map(tuple, next_point))
                                    if next_point is not None
                                    else None
                                ),
                                box=(
                                    tuple(next_box)
                                    if next_box is not None
                                    else None
                                ),
                            )
                        )

                        masks, ious, logits = predictor.predict(
                            point_coords=next_point,
                            point_labels=(
                                np.ones(1, dtype=int)
                                if next_point is not None
                                else None
                            ),
                            box=(
                                next_box[None, :]
                                if next_box is not None
                                else None
                            ),
                            mask_input=current_logits,
                            multimask_output=False,
                        )

                        current_mask = masks[0]
                        current_logits = logits
                        iou_score = float(ious[0])

                    else:
                        current_mask = np.zeros(
                            processed_frame.shape[:2],
                            dtype=bool,
                        )

                    time_seconds = self._resolve_frame_time(
                        config=config,
                        frame_index=frame_index,
                        video_fps=video_fps,
                    )

                    result = current_mask

                    records.append(
                        build_standard_stats_record(
                            source_path=config.input_path,
                            time_seconds=time_seconds,
                            frame_id=frame_index,
                            mask=current_mask,
                            mask_label=1,
                            iou_score=iou_score,
                            has_combined=config.save_combined,
                            mask_output_name=(
                                mask_output_path.name
                            ),
                            combined_output_name=(
                                combined_output_path.name
                                if config.save_combined
                                else None
                            ),
                        )
                    )

                else:
                    automatic_generator = (
                        automatic_mask_generator(predictor)
                    )

                    automatic_generator.initialize(
                        processed_frame,
                        verbose=False,
                    )

                    result = automatic_generator.generate()

                # The mask output must never contain prompts.
                mask_visualization = create_visualization(
                    processed_frame,
                    result,
                    prompts=None,
                    save_combined=False,
                    show_prompts=False,
                )

                outputs.append(
                    save_image(
                        mask_output_path,
                        mask_visualization,
                    )
                )

                if need_combined_frame:
                    combined_visualization = create_visualization(
                        processed_frame,
                        result,
                        prompts=(
                            current_prompt_overlay
                            if config.show_prompts
                            else None
                        ),
                        save_combined=True,
                        show_prompts=config.show_prompts,
                    )

                    if config.save_combined:
                        outputs.append(
                            save_image(
                                combined_output_path,
                                combined_visualization,
                            )
                        )

                    if config.save_combined_video:
                        combined_video_frames.append(
                            combined_visualization
                        )

                    if config.save_combined:
                        yield (
                            mask_visualization,
                            combined_visualization,
                        )
                    else:
                        yield mask_visualization

                else:
                    yield mask_visualization

                frame_count += 1

        for _ in frame_generator():
            pass

        self._validate_time_values(
            config=config,
            frame_count=frame_count,
        )

        if config.save_combined_video:
            combined_video_output = save_combined_video(
                output_dir,
                combined_video_frames,
                fps=video_fps,
            )
            outputs.append(combined_video_output)

        stats_path = (
            save_records(
                output_dir / "stats",
                records,
                config.export_format,
            )
            if records
            else None
        )

        if stats_path is not None:
            outputs.append(stats_path)

        return SegmentationResult(
            success=True,
            count=frame_count,
            message=f"Processed {frame_count} video frames.",
            outputs=tuple(outputs),
            stats_path=stats_path,
            metadata={
                "workflow": "video",
                "source_fps": source_fps,
                "output_fps": video_fps,
                "tracking": is_tracking,
                "tracking_method": (
                    config.tracking_method
                    if is_tracking
                    else None
                ),
                "frame_name_mode": config.frame_name_mode,
                "frame_name_prefix": config.frame_name_prefix,
            },
        )

    @staticmethod
    def _resolve_frame_time(
        *,
        config: WorkflowConfig,
        frame_index: int,
        video_fps: float,
    ) -> float:
        """Return the timestamp for one video frame."""

        if config.time_seconds is not None:
            if frame_index >= len(config.time_seconds):
                raise InputValidationError(
                    "Expected at least "
                    f"{frame_index + 1} time values, "
                    f"got {len(config.time_seconds)}."
                )

            return float(config.time_seconds[frame_index])

        return frame_index / video_fps

    @staticmethod
    def _validate_time_values(
        *,
        config: WorkflowConfig,
        frame_count: int,
    ) -> None:
        """Ensure a supplied time list matches the video frame count."""

        if (
            config.time_seconds is not None
            and len(config.time_seconds) != frame_count
        ):
            raise InputValidationError(
                f"Expected {frame_count} time values, "
                f"got {len(config.time_seconds)}."
            )

    @staticmethod
    def _next_prompt(
        mask: np.ndarray,
        method: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if method == "box":
            box = get_box_from_mask(
                mask,
                padding=20,
            )
            return None, box

        if method == "centroid":
            point = get_centroid(mask)
            return (
                np.asarray([point])
                if point is not None
                else None
            ), None

        point = get_pole_of_inaccessibility(mask)
        return (
            np.asarray([point])
            if point is not None
            else None
        ), None