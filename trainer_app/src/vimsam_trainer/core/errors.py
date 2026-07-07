class TrainerError(Exception):
    """Base error for user-facing trainer failures."""


class InputValidationError(TrainerError):
    """Raised when input paths, options, or file contents are invalid."""


class DependencyMissingError(TrainerError):
    """Raised when optional ML dependencies are required but unavailable."""


class OutputWriteError(TrainerError):
    """Raised when outputs cannot be written safely."""
