__all__ = ["read_u3cmos_raw", "get_raw_timeseries_files", "validate_raw_file_size", "validate_raw_timeseries_files"]


def __getattr__(name: str):
    if name in __all__:
        from . import raw

        return getattr(raw, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
