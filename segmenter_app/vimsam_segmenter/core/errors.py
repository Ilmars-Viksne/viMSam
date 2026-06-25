class SegmenterError(Exception):
    """Base error for user-facing segmenter failures."""


class InputValidationError(SegmenterError):
    """Raised when input paths, options, or file contents are invalid."""


class DependencyMissingError(SegmenterError):
    """Raised when optional ML dependencies are required but unavailable."""


class OutputWriteError(SegmenterError):
    """Raised when outputs cannot be written safely."""
