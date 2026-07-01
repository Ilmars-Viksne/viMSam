from .image_frames_logits import ImageFrameLogitsWorkflow
from .raw_single import RawSingleImageWorkflow
from .raw_timeseries import RawTimeSeriesWorkflow
from .raw_timeseries_logits import RawTimeSeriesLogitsWorkflow
from .single_image import SingleImageWorkflow
from .video_file import VideoFileWorkflow

WORKFLOW_MAP = {
    "single": SingleImageWorkflow,
    "video": VideoFileWorkflow,
    "raw_single": RawSingleImageWorkflow,
    "raw_timeseries": RawTimeSeriesWorkflow,
    "image_frames_logits": ImageFrameLogitsWorkflow,
    "raw_timeseries_logits": RawTimeSeriesLogitsWorkflow,
}

__all__ = [
    "WORKFLOW_MAP",
    "SingleImageWorkflow",
    "VideoFileWorkflow",
    "RawSingleImageWorkflow",
    "RawTimeSeriesWorkflow",
    "ImageFrameLogitsWorkflow",
    "RawTimeSeriesLogitsWorkflow",
]
