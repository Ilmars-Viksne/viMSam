from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..core.config import SegmentationResult, WorkflowConfig
from ..io.local import output_dir_for, save_image, save_records, save_video, write_video_streams
from ..io.raw import get_raw_timeseries_files, read_u3cmos_raw, validate_raw_timeseries_files
from ..processing.preprocess import PreProcessor
from ..utils.geometry import get_box_from_mask, get_centroid, get_pole_of_inaccessibility
from ..utils.logging import setup_logger
from ..utils.stats import StatsCollector
from ..utils.visualization import create_visualization

from .base import BaseWorkflow, automatic_mask_generator

logger = setup_logger("RawTimeSeriesWorkflow")


class RawTimeSeriesWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        files = get_raw_timeseries_files(config.input_path)
        if not files:
            return SegmentationResult(False, 0, "No raw files found.")
        validate_raw_timeseries_files(files, width=config.raw_width, height=config.raw_height)

        output_dir = output_dir_for(config.output_path)
        predictor = self.model_service.get_predictor()
        pre = PreProcessor()
        stats = StatsCollector()
        current_logits = None
        current_mask = None
        is_tracking = bool(config.prompts and config.prompts.points)
        points = np.array(config.prompts.points) if is_tracking else None
        outputs = []
        frame_count = 0

        logger.info("Starting raw time-series processing. Found %s frames. Tracking: %s", len(files), is_tracking)

        def frame_generator():
            nonlocal frame_count, current_logits, current_mask
            for i, filepath in tqdm(enumerate(files), desc="Processing Raw Series", total=len(files)):
                frame = read_u3cmos_raw(filepath, width=config.raw_width, height=config.raw_height)
                processed_frame = pre.run(frame)
                predictor.set_image(self.sam_image(processed_frame))
                prompt_viz = None
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
                        if config.show_prompts:
                            prompt_viz = {"type": "point", "data": points}
                    elif current_mask is not None and np.any(current_mask):
                        next_point, next_box = self._next_prompt(current_mask, config.tracking_method)
                        if config.show_prompts:
                            prompt_viz = {"type": "box", "data": next_box} if next_box is not None else {"type": "point", "data": next_point}
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
                    stats.collect(current_mask, iou, i, 1, meta={"filename": filepath.name})
                    result = current_mask
                else:
                    amg = automatic_mask_generator(predictor)
                    amg.initialize(processed_frame, verbose=False)
                    result = amg.generate()

                mask_viz = create_visualization(processed_frame, result, prompts=prompt_viz, save_combined=False)
                outputs.append(save_image(output_dir / f"frame_{i:05d}.png", mask_viz))

                if config.save_combined:
                    combined_viz = create_visualization(processed_frame, result, prompts=prompt_viz, save_combined=True)
                    outputs.append(save_image(output_dir / f"frame_{i:05d}_combined.png", combined_viz))
                    yield mask_viz, combined_viz
                else:
                    yield mask_viz

                frame_count += 1

        if config.save_combined:
            mask_path, combined_path = write_video_streams(
                output_dir / "result_video.mp4",
                output_dir / "result_video_combined.mp4",
                frame_generator(),
                5,
            )
            outputs.extend([mask_path, combined_path])
        else:
            outputs.append(save_video(output_dir / "result_video.mp4", frame_generator(), 5))

        stats_path = save_records(output_dir / "tracking_stats", stats.get_data(), config.export_format) if is_tracking else None
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
