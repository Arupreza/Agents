# src/artifact/__init__.py

# 1. Import base settings first
from .Settings import Parameter, GetVectorStore, _format_docs

# 2. Import logic that depends on settings
from .VectorSearch import BatchVectorSearch, Vector
from .WebSearch import BatchWebSearch, Web
from .Analysis import Analyze, BatchAnalyze, FormatReport

__all__ = [
    "Parameter",
    "GetVectorStore",
    "_format_docs",
    "BatchVectorSearch",
    "Vector",
    "BatchWebSearch",
    "Web",
    "Analyze",
    "BatchAnalyze",
    "FormatReport",
]