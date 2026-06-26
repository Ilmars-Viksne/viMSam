import numpy as np


class PreProcessor:
    def run(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif np.issubdtype(image.dtype, np.number) and image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)
