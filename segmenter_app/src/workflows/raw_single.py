import numpy as np
import os
from .base import BaseWorkflow
from src.core.config import WorkflowConfig, SegmentationResult
from src.core.io import IOFactory
from src.utils import setup_logger, StatsCollector, create_visualization
from src.processors import PreProcessor
from micro_sam.instance_segmentation import AutomaticMaskGenerator
from src.utils.raw_reader import read_u3cmos_raw

logger = setup_logger("RawSingleImageWorkflow")

class RawSingleImageWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        sink = IOFactory.get_sink(config.output_uri)

        # 1. Read Raw Image
        logger.info(f"Reading raw image: {config.input_uri}")
        image = read_u3cmos_raw(config.input_uri)

        # 2. Preprocess (Handles uint16 -> uint8 conversion automatically)
        pre = PreProcessor()
        processed = pre.run(image)

        if processed.ndim == 2:
            img_sam = np.stack((processed,) * 3, axis=-1)
        else:
            img_sam = processed

        predictor = self.model_service.get_predictor()
        predictor.set_image(img_sam)

        stats = StatsCollector()
        prompt_viz = None

        if config.prompts and config.prompts.points:
            points = np.array(config.prompts.points)
            if config.show_prompts: prompt_viz = {'type': 'point', 'data': points}

            masks_list =[]
            for i, pt in enumerate(points):
                masks, ious, _ = predictor.predict(
                    point_coords=np.array([pt]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )
                mask = masks[0]
                stats.collect(mask, float(ious[0]), 0, i+1)
                masks_list.append(mask)

            result = np.array(masks_list)
            sink.save_stats(stats.get_data(), "image_stats", config.export_format)
        else:
            amg = AutomaticMaskGenerator(predictor)
            amg.initialize(processed, verbose=False)
            result = amg.generate()

        # Determine Output Filename
        user_filename = os.path.basename(config.output_uri)
        user_ext = os.path.splitext(user_filename)[1]
        out_name = user_filename if user_ext else f"res_{os.path.splitext(os.path.basename(config.input_uri))[0]}.png"

        # Save Standard Mask
        mask_viz = create_visualization(processed, result, prompts=prompt_viz, save_combined=False)
        sink.save_image(mask_viz, out_name)

        # Save Combined View if requested
        if config.save_combined:
            combined_viz = create_visualization(processed, result, prompts=prompt_viz, save_combined=True)
            root, ext = os.path.splitext(out_name)
            sink.save_image(combined_viz, f"{root}_combined{ext}")

        return SegmentationResult(True, 1)
