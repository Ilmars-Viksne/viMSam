from __future__ import annotations

import numpy as np

from ..core.config import SegmentationResult, WorkflowConfig
from ..io.local import resolve_image_output, save_image, save_records, sibling_with_suffix
from ..io.raw import read_u3cmos_raw
from ..processing.preprocess import PreProcessor
from ..utils.logging import setup_logger
from ..utils.prompts import build_prompt_overlay
from ..utils.standard_stats import build_standard_stats_record
from ..utils.visualization import create_visualization

from .base import BaseWorkflow, automatic_mask_generator

logger = setup_logger("RawSingleImageWorkflow")


class RawSingleImageWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        logger.info("Reading raw image: %s", config.input_path)
        image = read_u3cmos_raw(config.input_path, width=config.raw_width, height=config.raw_height)
        processed = PreProcessor(method=config.preprocessing_method).run(image)
        predictor = self.model_service.get_predictor()
        predictor.set_image(self.sam_image(processed))

        records: list[dict[str, object]] = []
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
                records.append(
                    build_standard_stats_record(
                        source_path=config.input_path,
                        time_seconds=0.0,
                        frame_id=0,
                        mask=mask,
                        mask_label=i + 1,
                        iou_score=float(ious[0]),
                        has_combined=config.save_combined,
                    )
                )
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
                        image,
                        result,
                        prompts=current_prompt_overlay,
                        save_combined=True,
                        show_prompts=config.show_prompts,
                    ),
                )
            )

        if records:
            stats_path = save_records(out_path.parent / "image_stats", records, config.export_format)

        return SegmentationResult(True, 1, outputs=tuple(outputs), stats_path=stats_path)
