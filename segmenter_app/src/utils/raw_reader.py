import numpy as np
import os

#def read_u3cmos_raw(filepath, width=5440, height=3648):
def read_u3cmos_raw(filepath, width=1024, height=1024):
    """Reads a headerless 16-bit raw image file from the U3CMOS03100KPA sensor."""
    expected_size = width * height * 2

    with open(filepath, 'rb') as f:
        raw_data = f.read(expected_size)

    if len(raw_data) < expected_size:
        raise ValueError(f"File {filepath} is too small. Expected {expected_size} bytes, got {len(raw_data)}")

    # Map bytes to 16-bit unsigned integers
    image_1d = np.frombuffer(raw_data, dtype=np.uint16)
    image_2d = image_1d.reshape((height, width)).astype(np.float32)

    # --- FIX: Flip the image vertically ---
    image_2d = np.flipud(image_2d)

    # --- Percentile-based Contrast Normalization ---
    # Instead of a naive division by 256, we find the 1st and 99th percentiles.
    # This ignores extreme hot/dead pixels and stretches the actual cell data
    # to fill the visible 8-bit range (0-255), making it look like your reference image.
    p1, p99 = np.percentile(image_2d, (1, 99))

    # Clip extreme outliers
    image_2d = np.clip(image_2d, p1, p99)

    # Normalize to 0.0 - 255.0
    if p99 > p1:
        image_normalized = ((image_2d - p1) / (p99 - p1) * 255.0)
    else:
        image_normalized = np.zeros_like(image_2d)

    # Return as uint8.
    # Because it is now uint8, the existing PreProcessor will skip its default
    # division step, preserving the existing pipelines perfectly.
    return image_normalized.astype(np.uint8)

def get_raw_timeseries_files(directory):
    """Gets all files in a directory sorted alphabetically (for time-series)."""
    if not os.path.isdir(directory):
        raise ValueError(f"Input URI must be a directory for time-series, got {directory}")

    files = sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])
    return files
