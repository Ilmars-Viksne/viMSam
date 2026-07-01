from __future__ import annotations

import numpy as np

from ..core.config import SegmentationResult, WorkflowConfig
from ..io.local import load_image, resolve_image_output, save_image, save_records, sibling_with_suffix
from ..processing.preprocess import PreProcessor
from ..utils.prompts import build_prompt_overlay
from ..utils.stats import StatsCollector
from ..utils.visualization import create_visualization

from .base import BaseWorkflow, automatic_mask_generator


class SingleImageWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        image = load_image(config.input_path)
        processed = PreProcessor(method=config.preprocessing_method).run(image)
        predictor = self.model_service.get_predictor()
        predictor.set_image(self.sam_image(processed))

        stats = StatsCollector()
        stats_path = None
        current_prompt_overlay = build_prompt_overlay(
            points=config.prompts.points if config.prompts else None,
            box=config.prompts.box if config.prompts else None,
        )

        if config.prompts and config.prompts.points:
            points = np.array(config.prompts.points)
            masks_list = []
            for i, pt in enumerate(points):
                masks, ious, _ = predictor.predict(
                    point_coords=np.array([pt]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                mask = masks[0]
                stats.collect(mask, float(ious[0]), 0, i + 1)
                masks_list.append(mask)
            result = np.array(masks_list)
        else:
            amg = automatic_mask_generator(predictor)
            amg.initialize(processed, verbose=False)
            result = amg.generate()

        out_path = resolve_image_output(config.output_path, config.input_path)
        outputs = [
            save_image(
                out_path,
                create_visualization(
                    processed,
                    result,
                    prompts=None,
                    save_combined=False,
                    show_prompts=False,
                ),
            )
        ]

        if config.save_combined:
            combined_path = sibling_with_suffix(out_path, "_combined")
            outputs.append(
                save_image(
                    combined_path,
                    create_visualization(
                        processed,
                        result,
                        prompts=current_prompt_overlay,
                        save_combined=True,
                        show_prompts=config.show_prompts,
                    ),
                )
            )

        if stats.get_data():
            stats_path = save_records(out_path.parent / "image_stats", stats.get_data(), config.export_format)

        return SegmentationResult(True, 1, outputs=tuple(outputs), stats_path=stats_path)
