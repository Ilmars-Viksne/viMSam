from .logging import setup_logger

__all__ = [
    "setup_logger",
    "StatsCollector",
    "create_visualization",
    "save_masks_as_image",
    "get_centroid",
    "get_pole_of_inaccessibility",
    "get_box_from_mask",
]


def __getattr__(name: str):
    if name in {"get_box_from_mask", "get_centroid", "get_pole_of_inaccessibility"}:
        from . import geometry

        return getattr(geometry, name)
    if name == "StatsCollector":
        from .stats import StatsCollector

        return StatsCollector
    if name in {"create_visualization", "save_masks_as_image"}:
        from . import visualization

        return getattr(visualization, name)
    raise AttributeError(name)
