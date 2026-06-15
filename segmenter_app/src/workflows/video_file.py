import numpy as np
import os
from tqdm import tqdm
from .base import BaseWorkflow
from src.core.config import WorkflowConfig, SegmentationResult
from src.core.io import IOFactory
from src.utils import setup_logger, StatsCollector, get_box_from_mask, get_centroid, get_pole_of_inaccessibility, create_visualization
from src.processors import PreProcessor
from micro_sam.instance_segmentation import AutomaticMaskGenerator

logger = setup_logger("VideoWorkflow")

class VideoFileWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        source = IOFactory.get_source(config.input_uri)
        sink = IOFactory.get_sink(config.output_uri)

        meta = source.get_metadata()
        fps = meta.get('fps', 5)

        predictor = self.model_service.get_predictor()
        pre = PreProcessor()
        stats = StatsCollector()

        # State
        current_logits = None
        current_mask = None
        is_tracking = config.prompts is not None
        points = np.array(config.prompts.points) if (is_tracking and config.prompts.points) else None

        # We now track two sets of frames
        mask_frames = []
        combined_frames = []
        count = 0

        logger.info(f"Starting video processing. Tracking: {is_tracking}")

        for i, frame in tqdm(enumerate(source.stream_video()), desc="Processing"):
            processed_frame = pre.run(frame)

            # Prepare for SAM (H, W, 3)
            if processed_frame.ndim == 2:
                img_sam = np.stack((processed_frame,) * 3, axis=-1)
            else:
                img_sam = processed_frame

            predictor.set_image(img_sam)

            prompt_viz = None
            iou = 0.0

            if is_tracking:
                if i == 0:
                    # Initialize
                    masks, ious, logits = predictor.predict(
                        point_coords=points,
                        point_labels=np.ones(len(points), dtype=int),
                        multimask_output=False
                    )
                    current_mask = masks[0]
                    current_logits = logits
                    iou = float(ious[0])
                    if config.show_prompts: prompt_viz = {'type': 'point', 'data': points}
                else:
                    # Propagate
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

                        # Predict
                        masks, ious, logits = predictor.predict(
                            point_coords=next_point,\
                            point_labels=np.ones(1) if next_point is not None else None,
                            box=next_box[None, :] if next_box is not None else None,
                            mask_input=current_logits,
                            multimask_output=False
                        )
                        current_mask = masks[0]
                        current_logits = logits
                        iou = float(ious[0])
                    else:
                        current_mask = np.zeros(processed_frame.shape[:2], dtype=bool)

                # Collect Stats
                stats.collect(current_mask, iou, i, 1)
                result = current_mask
            else:
                # Auto
                amg = AutomaticMaskGenerator(predictor)
                amg.initialize(processed_frame, verbose=False)
                result = amg.generate()

            # --- 1. ALWAYS Generate & Save Raw Mask ---
            mask_viz = create_visualization(
                processed_frame,
                result,
                prompts=prompt_viz,
                save_combined=False # Force False for raw mask
            )
            fname_mask = f"frame_{i:05d}.png"
            sink.save_image(mask_viz, fname_mask)
            mask_frames.append(mask_viz)

            # --- 2. OPTIONALLY Generate & Save Combined ---
            if config.save_combined:
                combined_viz = create_visualization(
                    processed_frame,
                    result,
                    prompts=prompt_viz,
                    save_combined=True # Force True for combined
                )
                fname_combined = f"frame_{i:05d}_combined.png"
                sink.save_image(combined_viz, fname_combined)
                combined_frames.append(combined_viz)

            count += 1

        # --- Save Video Files ---

        # 1. Always save mask video
        sink.save_video(mask_frames, "result_video.mp4", fps)

        # 2. Optionally save combined video
        if config.save_combined and combined_frames:
            sink.save_video(combined_frames, "result_video_combined.mp4", fps)

        # Save Stats
        if is_tracking:
            sink.save_stats(stats.get_data(), "tracking_stats", config.export_format)

        return SegmentationResult(True, count)
