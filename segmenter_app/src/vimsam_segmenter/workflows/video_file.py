from __future__ import annotations

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
)
from ..io.video_outputs import save_combined_video
from ..processing.preprocess import PreProcessor
from ..utils.geometry import get_box_from_mask, get_centroid, get_pole_of_inaccessibility
from ..utils.logging import setup_logger
from ..utils.prompts import build_prompt_overlay
from ..utils.standard_stats import build_standard_stats_record
from ..utils.visualization import create_visualization

from .base import BaseWorkflow, automatic_mask_generator

logger = setup_logger("VideoWorkflow")


class VideoFileWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        output_dir = output_dir_for(config.output_path)
        fps = read_metadata(config.input_path).get("fps", 5)
        video_fps = config.fps or float(fps or 5)
        predictor = self.model_service.get_predictor()
        pre = PreProcessor(method=config.preprocessing_method)
        records: list[dict[str, object]] = []

        current_logits = None
        current_mask = None
        is_tracking = bool(config.prompts and config.prompts.points)
        points = np.array(config.prompts.points) if is_tracking else None
        outputs = []
        frame_count = 0
        combined_video_frames: list[np.ndarray] = []
        need_combined_frame = config.save_combined or config.save_combined_video
        ensure_series_output_dirs(output_dir, save_combined=config.save_combined)

        logger.info("Starting video processing. Tracking: %s", is_tracking)

        def frame_generator():
            nonlocal frame_count, current_logits, current_mask
            for i, frame in tqdm(enumerate(stream_video(config.input_path)), desc="Processing"):
                processed_frame = pre.run(frame)
                predictor.set_image(self.sam_image(processed_frame))
                current_prompt_overlay = {}
                iou = 0.0

                if is_tracking:
                    if i == 0:
                        masks, ious, logits = predictor.predict(
                            point_coords=points,
                            point_labels=np.ones(len(points), dtype=int),
                            multimask_output=False,
                        )
                        current_mask = masks[0]
                        current_logits = logits
                        iou = float(ious[0])
                        current_prompt_overlay = build_prompt_overlay(
                            points=tuple(points) if points is not None else None,
                            box=None,
                        )
                    elif current_mask is not None and np.any(current_mask):
                        next_point, next_box = self._next_prompt(current_mask, config.tracking_method)
                        current_prompt_overlay = build_prompt_overlay(
                            points=tuple(map(tuple, next_point)) if next_point is not None else None,
                            box=tuple(next_box) if next_box is not None else None,
                        )
                        masks, ious, logits = predictor.predict(
                            point_coords=next_point,
                            point_labels=np.ones(1) if next_point is not None else None,
                            box=next_box[None, :] if next_box is not None else None,
                            mask_input=current_logits,
                            multimask_output=False,
                        )
                        current_mask = masks[0]
                        current_logits = logits
                        iou = float(ious[0])
                    else:
                        current_mask = np.zeros(processed_frame.shape[:2], dtype=bool)
                    if config.time_seconds is not None:
                        if i >= len(config.time_seconds):
                            raise InputValidationError(
                                f"Expected at least {i + 1} time values, got {len(config.time_seconds)}."
                            )
                        time_seconds = config.time_seconds[i]
                    else:
                        time_seconds = i / video_fps
                    records.append(
                        build_standard_stats_record(
                            source_path=config.input_path,
                            time_seconds=time_seconds,
                            frame_id=i,
                            mask=current_mask,
                            mask_label=1,
                            iou_score=iou,
                            has_combined=config.save_combined,
                        )
                    )
                    result = current_mask
                else:
                    amg = automatic_mask_generator(predictor)
                    amg.initialize(processed_frame, verbose=False)
                    result = amg.generate()

                mask_viz = create_visualization(
                    processed_frame,
                    result,
                    prompts=None,
                    save_combined=False,
                    show_prompts=False,
                )
                outputs.append(save_image(mask_frame_path(output_dir, i), mask_viz))

                if need_combined_frame:
                    combined_viz = create_visualization(
                        processed_frame,
                        result,
                        prompts=current_prompt_overlay if config.show_prompts else None,
                        save_combined=True,
                        show_prompts=config.show_prompts,
                    )
                    if config.save_combined:
                        outputs.append(save_image(combined_frame_path(output_dir, i), combined_viz))
                    if config.save_combined_video:
                        combined_video_frames.append(combined_viz)
                    if config.save_combined:
                        yield mask_viz, combined_viz
                    else:
                        yield mask_viz
                else:
                    yield mask_viz

                frame_count += 1

        for _ in frame_generator():
            pass

        if config.time_seconds is not None and len(config.time_seconds) != frame_count:
            raise InputValidationError(f"Expected {frame_count} time values, got {len(config.time_seconds)}.")

        if config.save_combined_video:
            outputs.append(save_combined_video(output_dir, combined_video_frames, fps=config.fps))

        stats_path = save_records(output_dir / "stats", records, config.export_format) if records else None
        return SegmentationResult(True, frame_count, outputs=tuple(outputs), stats_path=stats_path)

    @staticmethod
    def _next_prompt(mask: np.ndarray, method: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        if method == "box":
            return None, get_box_from_mask(mask, padding=20)
        if method == "centroid":
            pt = get_centroid(mask)
            return (np.array([pt]) if pt else None), None
        pt = get_pole_of_inaccessibility(mask)
        return (np.array([pt]) if pt else None), None
