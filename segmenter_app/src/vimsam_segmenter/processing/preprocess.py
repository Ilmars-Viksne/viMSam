import numpy as np


class PreProcessor:
    def __init__(self, method: str = "fixed_16bit") -> None:
        self.method = self._normalize_method(method)

    @staticmethod
    def _normalize_method(method: str) -> str:
        normalized = str(method).strip().lower()
        if normalized not in {"fixed_16bit", "minmax", "percentile", "none"}:
            raise ValueError("preprocessing method must be one of: fixed_16bit, minmax, percentile, none")
        return normalized

    def run(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)

        if self.method == "fixed_16bit" and image.dtype == np.uint16:
            return np.ascontiguousarray((image / 256).astype(np.uint8))

        if self.method == "percentile":
            if not np.issubdtype(image.dtype, np.number):
                return np.ascontiguousarray(image)
            low, high = np.percentile(image, (1, 99))
            if high <= low:
                return np.zeros_like(image, dtype=np.uint8)
            scaled = (image - low) / (high - low)
            return np.ascontiguousarray((np.clip(scaled, 0, 1) * 255).astype(np.uint8))

        if self.method == "minmax":
            if not np.issubdtype(image.dtype, np.number):
                return np.ascontiguousarray(image)
            mn, mx = float(image.min()), float(image.max())
            if mx <= mn:
                return np.zeros_like(image, dtype=np.uint8)
            return np.ascontiguousarray(((image - mn) / (mx - mn) * 255).astype(np.uint8))

        if self.method == "none":
            return np.ascontiguousarray(image)

        if np.issubdtype(image.dtype, np.number):
            return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
        return np.ascontiguousarray(image)
