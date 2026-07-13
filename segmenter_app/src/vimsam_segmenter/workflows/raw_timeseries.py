from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from ..core.config import SegmentationResult, WorkflowConfig
from ..io.local import output_dir_for, save_image, save_records
from ..io.raw import (
    get_raw_timeseries_files,
    read_u3cmos_raw,
    validate_raw_timeseries_files,
)
from ..io.series_outputs import (
    combined_frame_path,
    ensure_series_output_dirs,
    mask_frame_path,
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
from ..utils.time_resolver import resolve_time_seconds
from ..utils.visualization import create_visualization

from .base import BaseWorkflow, automatic_mask_generator


logger = setup_logger("RawTimeSeriesWorkflow")


class RawTimeSeriesWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        files = get_raw_timeseries_files(config.input_path)

        if not files:
            return SegmentationResult(
                success=False,
                count=0,
                message="No raw files found.",
            )

        validate_raw_timeseries_files(
            files,
            width=config.raw_width,
            height=config.raw_height,
        )

        output_dir = output_dir_for(config.output_path)
        predictor = self.model_service.get_predictor()
        preprocessor = PreProcessor(
            method=config.preprocessing_method,
        )

        times = resolve_time_seconds(
            source_paths=files,
            fps=config.fps,
            timestamp_format=config.timestamp_format,
            user_time_seconds=config.time_seconds,
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
            "Starting raw time-series processing. "
            "Found %s frames. Tracking: %s",
            len(files),
            is_tracking,
        )

        def frame_generator() -> Iterator[
            np.ndarray | tuple[np.ndarray, np.ndarray]
        ]:
            nonlocal frame_count
            nonlocal current_logits
            nonlocal current_mask

            indexed_files = enumerate(files)

            for frame_index, source_path in tqdm(
                indexed_files,
                desc="Processing Raw Series",
                total=len(files),
            ):
                # Resolve output filenames before statistics are created.
                mask_output_path = mask_frame_path(
                    output_dir,
                    frame_index=frame_index,
                    mode=config.frame_name_mode,
                    source_path=source_path,
                    prefix=config.frame_name_prefix,
                )

                combined_output_path = combined_frame_path(
                    output_dir,
                    frame_index=frame_index,
                    mode=config.frame_name_mode,
                    source_path=source_path,
                    prefix=config.frame_name_prefix,
                )

                raw_frame = read_u3cmos_raw(
                    source_path,
                    width=config.raw_width,
                    height=config.raw_height,
                )

                processed_frame = preprocessor.run(raw_frame)
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

                    result = current_mask

                    records.append(
                        build_standard_stats_record(
                            source_path=source_path,
                            time_seconds=times[frame_index],
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

                # Clean mask frame: prompts are never drawn here.
                mask_viz = create_visualization(
                    processed_frame,
                    result,
                    prompts=None,
                    save_combined=False,
                    show_prompts=False,
                )

                outputs.append(
                    save_image(
                        mask_output_path,
                        mask_viz,
                    )
                )

                if need_combined_frame:
                    combined_viz = create_visualization(
                        raw_frame,
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
                                combined_viz,
                            )
                        )

                    if config.save_combined_video:
                        combined_video_frames.append(
                            combined_viz
                        )

                    if config.save_combined:
                        yield mask_viz, combined_viz
                    else:
                        yield mask_viz

                else:
                    yield mask_viz

                frame_count += 1

        for _ in frame_generator():
            pass

        if config.save_combined_video:
            combined_video_output = save_combined_video(
                output_dir,
                combined_video_frames,
                fps=config.fps,
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
            message=(
                f"Processed {frame_count} raw time-series frames."
            ),
            outputs=tuple(outputs),
            stats_path=stats_path,
            metadata={
                "workflow": "raw_timeseries",
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