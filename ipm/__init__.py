"""IMP302m – Phân đoạn, đếm và phân loại kích thước vật thể bằng xử lý ảnh truyền thống."""
from .config import Config
from .pipeline import PipelineResult, run

__all__ = ["Config", "PipelineResult", "run"]
