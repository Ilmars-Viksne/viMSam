import numpy as np
class PreProcessor:
    def run(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            image = (image / 256).astype('uint8')
        image = np.ascontiguousarray(image)
        return image
