from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from vimsam_segmenter.utils.geometry import (
    get_box_from_mask,
    get_centroid,
    get_pole_of_inaccessibility,
)


PromptFallback = Literal["centroid", "pole", "box", "none"]


@dataclass(slots=True)
class LogitPropagationState:
    """Stores temporal state for SAM-style mask-input propagation."""

    low_res_logits: np.ndarray | None = None
    mask: np.ndarray | None = None
    score: float | None = None


@dataclass(slots=True)
class LogitPropagationResult:
    mask: np.ndarray
    low_res_logits: np.ndarray | None
    score: float | None
    used_mask_input: bool
    used_fallback_prompt: bool


class LogitPropagationTracker:
    """SAM-like temporal propagation helper that uses previous logits as mask_input."""

    def __init__(
        self,
        *,
        fallback: PromptFallback = "pole",
        point_label: int = 1,
        box_padding: int = 20,
        reset_on_empty_mask: bool = True,
    ) -> None:
        self.fallback = fallback
        self.point_label = point_label
        self.box_padding = box_padding
        self.reset_on_empty_mask = reset_on_empty_mask
        self.state = LogitPropagationState()

    def reset(self) -> None:
        self.state = LogitPropagationState()

    def initialize_from_prompt(
        self,
        *,
        predictor,
        points: tuple[tuple[int, int], ...] | None = None,
        box: tuple[int, int, int, int] | None = None,
    ) -> LogitPropagationResult:
        point_coords = None
        point_labels = None
        box_array = None

        if points:
            point_coords = np.asarray(points, dtype=np.float32)
            point_labels = np.full((len(points),), self.point_label, dtype=np.int32)

        if box is not None:
            box_array = np.asarray(box, dtype=np.float32)

        result = self._predict_single_mask(
            predictor=predictor,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_array,
            mask_input=None,
        )

        self._update_state(result)
        return result

    def propagate(self, *, predictor) -> LogitPropagationResult:
        mask_input = self._prepare_mask_input(self.state.low_res_logits)

        point_coords = None
        point_labels = None
        box = None
        used_fallback_prompt = False

        if self.state.mask is not None and self.fallback != "none":
            point_coords, point_labels, box = self._make_fallback_prompt(self.state.mask)
            used_fallback_prompt = (
                point_coords is not None or point_labels is not None or box is not None
            )

        result = self._predict_single_mask(
            predictor=predictor,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
        )

        result.used_fallback_prompt = used_fallback_prompt
        self._update_state(result)
        return result

    def _make_fallback_prompt(
        self,
        mask: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self.fallback == "centroid":
            point = get_centroid(mask)
            if point is None:
                return None, None, None
            return (
                np.asarray([point], dtype=np.float32),
                np.asarray([self.point_label], dtype=np.int32),
                None,
            )

        if self.fallback == "pole":
            point = get_pole_of_inaccessibility(mask)
            if point is None:
                return None, None, None
            return (
                np.asarray([point], dtype=np.float32),
                np.asarray([self.point_label], dtype=np.int32),
                None,
            )

        if self.fallback == "box":
            box = get_box_from_mask(mask, padding=self.box_padding)
            if box is None:
                return None, None, None
            return None, None, box.astype(np.float32)

        return None, None, None

    def _prepare_mask_input(self, logits: np.ndarray | None) -> np.ndarray | None:
        if logits is None:
            return None

        arr = np.asarray(logits)

        if arr.ndim == 4:
            arr = arr[0]

        if arr.ndim == 3:
            arr = arr[0]

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D low-res logits, got shape {arr.shape}")

        return arr[None, :, :].astype(np.float32)

    def _predict_single_mask(
        self,
        *,
        predictor,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: np.ndarray | None,
        mask_input: np.ndarray | None,
    ) -> LogitPropagationResult:
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=False,
        )

        masks = np.asarray(masks)
        scores = np.asarray(scores) if scores is not None else None
        logits = np.asarray(logits) if logits is not None else None

        if masks.ndim == 3:
            mask = masks[0]
        elif masks.ndim == 2:
            mask = masks
        else:
            raise ValueError(f"Unexpected mask shape returned by predictor: {masks.shape}")

        score = None
        if scores is not None and scores.size > 0:
            score = float(scores.reshape(-1)[0])

        low_res_logits = None
        if logits is not None:
            if logits.ndim == 4:
                low_res_logits = logits[0, 0]
            elif logits.ndim == 3:
                low_res_logits = logits[0]
            elif logits.ndim == 2:
                low_res_logits = logits
            else:
                raise ValueError(f"Unexpected logits shape returned by predictor: {logits.shape}")

        return LogitPropagationResult(
            mask=mask.astype(bool),
            low_res_logits=low_res_logits,
            score=score,
            used_mask_input=mask_input is not None,
            used_fallback_prompt=False,
        )

    def _update_state(self, result: LogitPropagationResult) -> None:
        mask_has_pixels = bool(np.any(result.mask))

        if not mask_has_pixels and self.reset_on_empty_mask:
            self.reset()
            return

        self.state = LogitPropagationState(
            low_res_logits=result.low_res_logits,
            mask=result.mask,
            score=result.score,
        )
