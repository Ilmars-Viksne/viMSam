import numpy as np
import os
from .base import BaseWorkflow
from src.core.config import WorkflowConfig, SegmentationResult
from src.core.io import IOFactory
from src.utils import setup_logger, StatsCollector, create_visualization
from src.processors import PreProcessor
from micro_sam.instance_segmentation import AutomaticMaskGenerator

logger = setup_logger("SingleImageWorkflow")

class SingleImageWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        source = IOFactory.get_source(config.input_uri)
        sink = IOFactory.get_sink(config.output_uri)

        image = source.load_image()
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

            masks_list = []
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

        # --- Determine Output Filename ---
        user_filename = os.path.basename(config.output_uri)
        user_ext = os.path.splitext(user_filename)[1]

        if user_ext:
            out_name = user_filename
        else:
            fname = os.path.basename(config.input_uri)
            name_no_ext = os.path.splitext(fname)[0]
            out_name = f"res_{name_no_ext}.png"

        # --- 1. ALWAYS Save Raw Mask ---
        mask_viz = create_visualization(
            processed,
            result,
            prompts=prompt_viz,
            save_combined=False
        )
        sink.save_image(mask_viz, out_name)

        # --- 2. OPTIONALLY Save Combined ---
        if config.save_combined:
            combined_viz = create_visualization(
                processed,
                result,
                prompts=prompt_viz,
                save_combined=True
            )
            # Create suffix name: res_image.png -> res_image_combined.png
            root, ext = os.path.splitext(out_name)
            out_name_combined = f"{root}_combined{ext}"
            sink.save_image(combined_viz, out_name_combined)

        return SegmentationResult(True, 1)
