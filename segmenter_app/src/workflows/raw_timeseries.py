import numpy as np
import os
from tqdm import tqdm
from .base import BaseWorkflow
from src.core.config import WorkflowConfig, SegmentationResult
from src.core.io import IOFactory
from src.utils import setup_logger, StatsCollector, get_box_from_mask, get_centroid, get_pole_of_inaccessibility, create_visualization
from src.processors import PreProcessor
from micro_sam.instance_segmentation import AutomaticMaskGenerator
from src.utils.raw_reader import read_u3cmos_raw, get_raw_timeseries_files

logger = setup_logger("RawTimeSeriesWorkflow")

class RawTimeSeriesWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        # Input URI is expected to be a directory containing raw files
        files = get_raw_timeseries_files(config.input_uri)
        if not files:
            logger.error(f"No files found in {config.input_uri}")
            return SegmentationResult(False, 0, "No raw files found.")

        sink = IOFactory.get_sink(config.output_uri)
        fps = 5 # Default FPS for output video

        predictor = self.model_service.get_predictor()
        pre = PreProcessor()
        stats = StatsCollector()

        current_logits = None
        current_mask = None
        is_tracking = config.prompts is not None
        points = np.array(config.prompts.points) if (is_tracking and config.prompts.points) else None

        mask_frames = []
        combined_frames =[]
        count = 0

        logger.info(f"Starting raw time-series processing. Found {len(files)} frames. Tracking: {is_tracking}")

        for i, filepath in tqdm(enumerate(files), desc="Processing Raw Series", total=len(files)):
            # Read and preprocess raw frame
            frame = read_u3cmos_raw(filepath)
            processed_frame = pre.run(frame)

            img_sam = np.stack((processed_frame,) * 3, axis=-1) if processed_frame.ndim == 2 else processed_frame
            predictor.set_image(img_sam)

            prompt_viz = None
            iou = 0.0

            if is_tracking:
                if i == 0:
                    masks, ious, logits = predictor.predict(
                        point_coords=points, point_labels=np.ones(len(points), dtype=int), multimask_output=False
                    )
                    current_mask, current_logits, iou = masks[0], logits, float(ious[0])
                    if config.show_prompts: prompt_viz = {'type': 'point', 'data': points}
                else:
                    if current_mask is not None and np.any(current_mask):
                        next_point, next_box = None, None
                        if config.tracking_method == 'box':
                            next_box = get_box_from_mask(current_mask, padding=20)
                            if config.show_prompts: prompt_viz = {'type': 'box', 'data': next_box}
                        elif config.tracking_method == 'centroid':
                            pt = get_centroid(current_mask)
                            if pt: next_point = np.array([pt])
                            if config.show_prompts: prompt_viz = {'type': 'point', 'data': next_point}
                        elif config.tracking_method == 'pole':
                            pt = get_pole_of_inaccessibility(current_mask)
                            if pt: next_point = np.array([pt])
                            if config.show_prompts: prompt_viz = {'type': 'point', 'data': next_point}

                        masks, ious, logits = predictor.predict(
                            point_coords=next_point, point_labels=np.ones(1) if next_point is not None else None,
                            box=next_box[None, :] if next_box is not None else None, mask_input=current_logits, multimask_output=False
                        )
                        current_mask, current_logits, iou = masks[0], logits, float(ious[0])
                    else:
                        current_mask = np.zeros(processed_frame.shape[:2], dtype=bool)

                stats.collect(current_mask, iou, i, 1, meta={"filename": os.path.basename(filepath)})
                result = current_mask
            else:
                amg = AutomaticMaskGenerator(predictor)
                amg.initialize(processed_frame, verbose=False)
                result = amg.generate()

            # Save Visualizations
            mask_viz = create_visualization(processed_frame, result, prompts=prompt_viz, save_combined=False)
            sink.save_image(mask_viz, f"frame_{i:05d}.png")
            mask_frames.append(mask_viz)

            if config.save_combined:
                combined_viz = create_visualization(processed_frame, result, prompts=prompt_viz, save_combined=True)
                sink.save_image(combined_viz, f"frame_{i:05d}_combined.png")
                combined_frames.append(combined_viz)

            count += 1

        # Compile Output Videos
        sink.save_video(mask_frames, "result_video.mp4", fps)
        if config.save_combined and combined_frames:
            sink.save_video(combined_frames, "result_video_combined.mp4", fps)

        if is_tracking:
            sink.save_stats(stats.get_data(), "tracking_stats", config.export_format)

        return SegmentationResult(True, count)
