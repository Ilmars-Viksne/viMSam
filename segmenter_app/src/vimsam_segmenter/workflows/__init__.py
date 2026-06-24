from .raw_single import RawSingleImageWorkflow
from .raw_timeseries import RawTimeSeriesWorkflow
from .single_image import SingleImageWorkflow
from .video_file import VideoFileWorkflow

WORKFLOW_MAP = {
    "single": SingleImageWorkflow,
    "video": VideoFileWorkflow,
    "raw_single": RawSingleImageWorkflow,
    "raw_timeseries": RawTimeSeriesWorkflow,
}

__all__ = ["WORKFLOW_MAP", "SingleImageWorkflow", "VideoFileWorkflow", "RawSingleImageWorkflow", "RawTimeSeriesWorkflow"]
