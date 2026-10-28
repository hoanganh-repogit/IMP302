"""IMP302m – Segmentation, counting and size classification of objects with classical image processing."""
from .config import Config
from .pipeline import PipelineResult, run

__all__ = ["Config", "PipelineResult", "run"]
