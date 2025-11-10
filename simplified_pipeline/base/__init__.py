"""
Base abstract classes for the simplified pipeline.

This module provides abstract base classes that define interfaces
for core components, making it easy to swap implementations and
experiment with different approaches.
"""

from .chunker_base import ChunkerBase
from .extractor_base import ExtractorBase
from .embedder_base import EmbedderBase

__all__ = [
    "ChunkerBase",
    "ExtractorBase",
    "EmbedderBase",
]
