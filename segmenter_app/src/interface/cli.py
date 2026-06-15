import argparse
import numpy as np
from src.core.config import WorkflowConfig, PromptConfig, ModelConfig
from src.core.model_service import ModelService
from src.workflows import WORKFLOW_MAP
from src.utils import setup_logger

logger = setup_logger("CLI")

class CLI:
    def _parse_points(self, points_str):
        if not points_str: return None
        try:
            point_list =[]
            groups = points_str.strip().split(' ')
            for group in groups:
                if ',' not in group: continue
                x, y = map(int, group.split(','))
                point_list.append([x, y])
            return point_list
        except:
            return None

    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True, help="Input file or directory (for time-series)")
        parser.add_argument("--out", required=True)
        # Added new workflow choices here
        parser.add_argument("--workflow", choices=["single", "video", "raw_single", "raw_timeseries"], default="single")
        parser.add_argument("--model", default="vit_b")
        parser.add_argument("--points", type=str, default=None)
        parser.add_argument("--tracking_method", default="box")
        parser.add_argument("--format", default="csv")

        parser.add_argument("--show_prompts", action="store_true", help="Draw prompts on the output (overlay).")
        parser.add_argument("--save_combined", action="store_true", help="Save 3-panel image (Original|Mask|Overlay) instead of just Mask.")

        import sys
        if len(sys.argv) == 1:
            parser.print_help()
            return

        args = parser.parse_args()

        p_config = None
        if args.points:
            pts = self._parse_points(args.points)
            if pts: p_config = PromptConfig(points=pts)

        config = WorkflowConfig(
            workflow_type=args.workflow,
            input_uri=args.input,
            output_uri=args.out,
            model=ModelConfig(name=args.model),
            prompts=p_config,
            show_prompts=args.show_prompts,
            save_combined=args.save_combined,
            tracking_method=args.tracking_method,
            export_format=args.format
        )

        service = ModelService()
        service.load_model(config.model.name)

        wf_class = WORKFLOW_MAP.get(config.workflow_type)
        if not wf_class:
            logger.error(f"Workflow {config.workflow_type} not found!")
            return

        wf = wf_class(service)
        result = wf.run(config)
        logger.info(f"Result: {result}")
