__all__ = ["PreProcessor"]


def __getattr__(name: str):
    if name == "PreProcessor":
        from .preprocess import PreProcessor

        return PreProcessor
    raise AttributeError(name)
