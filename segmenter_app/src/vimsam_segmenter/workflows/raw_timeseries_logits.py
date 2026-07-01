from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..core.config import SegmentationResult, WorkflowConfig
from ..io.local import output_dir_for, save_image, save_records
from ..io.raw import get_raw_timeseries_files, read_u3cmos_raw, validate_raw_timeseries_files
from ..processing.preprocess import PreProcessor
from ..tracking.logit_propagation import LogitPropagationTracker
from ..utils.geometry import get_box_from_mask, get_centroid, get_pole_of_inaccessibility
from ..utils.logging import setup_logger
from ..utils.visualization import create_visualization

from .base import BaseWorkflow


logger = setup_logger("RawTimeSeriesLogitsWorkflow")


class RawTimeSeriesLogitsWorkflow(BaseWorkflow):
    def run(self, config: WorkflowConfig) -> SegmentationResult:
        output_dir = output_dir_for(config.output_path)

        raw_files = get_raw_timeseries_files(config.input_path)
        if not raw_files:
            return SegmentationResult(
                success=False,
                count=0,
                message=f"No raw frames found in {config.input_path}",
            )

        validate_raw_timeseries_files(raw_files, width=config.raw_width, height=config.raw_height)

        predictor = self.model_service.get_predictor()
        preprocessor = PreProcessor(method=config.preprocessing_method)

        tracker = LogitPropagationTracker(
            fallback=config.tracking_method,
            reset_on_empty_mask=True,
        )

        records: list[dict[str, object]] = []
        outputs: list[Path] = []

        prompt_points = None
        prompt_box = None

        if config.prompts is not None:
            prompt_points = config.prompts.points
            prompt_box = config.prompts.box

        if not prompt_points and prompt_box is None:
            return SegmentationResult(
                success=False,
                count=0,
                message=(
                    "The first frame of raw_timeseries_logits requires "
                    "at least one point prompt or a box prompt."
                ),
            )

        fps = self._fps_from_config_metadata(config)
        previous_centroid: tuple[int, int] | None = None
        previous_area: int | None = None

        for frame_index, raw_path in enumerate(tqdm(raw_files, desc="Raw logit propagation")):
            raw_image = read_u3cmos_raw(raw_path, width=config.raw_width, height=config.raw_height)
            processed = preprocessor.run(raw_image)

            predictor.set_image(self.sam_image(processed))

            if frame_index == 0:
                result = tracker.initialize_from_prompt(
                    predictor=predictor,
                    points=prompt_points,
                    box=prompt_box,
                )
            else:
                result = tracker.propagate(predictor=predictor)

            mask = result.mask

            mask_output_path = output_dir / f"{raw_path.stem}_mask.png"
            combined_output_path = output_dir / f"{raw_path.stem}_combined.png"

            mask_image = mask.astype(np.uint8) * 255
            save_image(mask_output_path, mask_image)
            outputs.append(mask_output_path)

            if config.save_combined:
                prompts = self._prompt_visualization_payload(
                    points=prompt_points if frame_index == 0 else None,
                    box=prompt_box if frame_index == 0 else None,
                    mask=mask,
                    tracking_method=config.tracking_method,
                    show_prompts=config.show_prompts,
                )

                combined = create_visualization(
                    processed,
                    mask,
                    prompts=prompts,
                    save_combined=True,
                )
                save_image(combined_output_path, combined)
                outputs.append(combined_output_path)

            record = self._record_frame(
                frame_index=frame_index,
                raw_path=raw_path,
                mask=mask,
                score=result.score,
                used_mask_input=result.used_mask_input,
                used_fallback_prompt=result.used_fallback_prompt,
                tracking_method=config.tracking_method,
                fps=fps,
                previous_centroid=previous_centroid,
                previous_area=previous_area,
            )

            records.append(record)

            previous_centroid = get_centroid(mask)
            previous_area = int(np.sum(mask))

        stats_path = save_records(
            output_dir / "raw_timeseries_logits_stats",
            records,
            config.export_format,
        )

        if stats_path is not None:
            outputs.append(stats_path)

        return SegmentationResult(
            success=True,
            count=len(raw_files),
            message=(
                f"Processed {len(raw_files)} raw frames using "
                "logits-based mask-input propagation."
            ),
            outputs=tuple(outputs),
            stats_path=stats_path,
            metadata={
                "workflow": "raw_timeseries_logits",
                "tracking_method": config.tracking_method,
                "used_logits_propagation": True,
                "raw_width": config.raw_width,
                "raw_height": config.raw_height,
            },
        )

    def sam_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)

        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            return np.repeat(arr, 3, axis=-1)

        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr

        if arr.ndim == 3 and arr.shape[-1] == 4:
            return arr[..., :3]

        raise ValueError(f"Unsupported image shape for SAM input: {arr.shape}")

    def _fps_from_config_metadata(self, config: WorkflowConfig) -> float | None:
        fps = getattr(config, "fps", None)
        if fps is not None and fps > 0:
            return float(fps)
        return None

    def _record_frame(
        self,
        *,
        frame_index: int,
        raw_path: Path,
        mask: np.ndarray,
        score: float | None,
        used_mask_input: bool,
        used_fallback_prompt: bool,
        tracking_method: str,
        fps: float | None,
        previous_centroid: tuple[int, int] | None,
        previous_area: int | None,
    ) -> dict[str, object]:
        centroid = get_centroid(mask)
        pole = get_pole_of_inaccessibility(mask)
        box = get_box_from_mask(mask, padding=0)

        area_px = int(np.sum(mask))
        time_seconds = None
        if fps is not None and fps > 0:
            time_seconds = frame_index / fps

        centroid_displacement_px = None
        velocity_px_per_s = None

        if previous_centroid is not None and centroid is not None:
            dx = centroid[0] - previous_centroid[0]
            dy = centroid[1] - previous_centroid[1]
            centroid_displacement_px = float(np.sqrt(dx * dx + dy * dy))

            if fps is not None and fps > 0:
                velocity_px_per_s = centroid_displacement_px * fps

        area_change_px = None
        area_change_fraction = None

        if previous_area is not None:
            area_change_px = area_px - previous_area
            if previous_area > 0:
                area_change_fraction = area_change_px / previous_area

        record: dict[str, object] = {
            "frame_index": frame_index,
            "time_seconds": time_seconds,
            "source": str(raw_path),
            "area_px": area_px,
            "area_change_px": area_change_px,
            "area_change_fraction": area_change_fraction,
            "centroid_displacement_px": centroid_displacement_px,
            "velocity_px_per_s": velocity_px_per_s,
            "sam_score": score,
            "used_mask_input": used_mask_input,
            "used_fallback_prompt": used_fallback_prompt,
            "tracking_method": tracking_method,
        }

        if centroid is not None:
            record["centroid_x"] = centroid[0]
            record["centroid_y"] = centroid[1]
        else:
            record["centroid_x"] = None
            record["centroid_y"] = None

        if pole is not None:
            record["pole_x"] = pole[0]
            record["pole_y"] = pole[1]
        else:
            record["pole_x"] = None
            record["pole_y"] = None

        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            record["bbox_x1"] = x1
            record["bbox_y1"] = y1
            record["bbox_x2"] = x2
            record["bbox_y2"] = y2
            record["bbox_width"] = x2 - x1
            record["bbox_height"] = y2 - y1
        else:
            record["bbox_x1"] = None
            record["bbox_y1"] = None
            record["bbox_x2"] = None
            record["bbox_y2"] = None
            record["bbox_width"] = None
            record["bbox_height"] = None

        return record

    def _prompt_visualization_payload(
        self,
        *,
        points: tuple[tuple[int, int], ...] | None,
        box: tuple[int, int, int, int] | None,
        mask: np.ndarray,
        tracking_method: str,
        show_prompts: bool,
    ) -> dict[str, np.ndarray] | None:
        if not show_prompts:
            return None

        prompts: dict[str, np.ndarray] = {}

        if points:
            prompts["points"] = np.asarray(points, dtype=np.float32)

        if box is not None:
            prompts["box"] = np.asarray(box, dtype=np.float32)

        if not prompts:
            if tracking_method == "centroid":
                point = get_centroid(mask)
                if point is not None:
                    prompts["points"] = np.asarray([point], dtype=np.float32)
            elif tracking_method == "pole":
                point = get_pole_of_inaccessibility(mask)
                if point is not None:
                    prompts["points"] = np.asarray([point], dtype=np.float32)
            elif tracking_method == "box":
                derived_box = get_box_from_mask(mask)
                if derived_box is not None:
                    prompts["box"] = derived_box.astype(np.float32)

        return prompts or None
