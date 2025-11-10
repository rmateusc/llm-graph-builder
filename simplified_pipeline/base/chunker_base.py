"""
Abstract base class for document chunkers.

Defines the interface for all chunking strategies (token-based, character-based, semantic, etc.)
"""

from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document


class ChunkerBase(ABC):
    """Abstract base class for document chunking strategies"""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initialize chunker with size and overlap parameters

        Args:
            chunk_size: Size of each chunk (tokens or characters depending on implementation)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of LangChain Document objects to chunk

        Returns:
            List of chunked Document objects with preserved metadata
        """
        pass

    @abstractmethod
    def get_chunk_count_estimate(self, documents: List[Document]) -> int:
        """
        Estimate the number of chunks that will be created

        Args:
            documents: List of documents to estimate for

        Returns:
            Estimated number of chunks
        """
        pass

    def get_chunker_info(self) -> dict:
        """
        Get information about this chunker for experiment tracking

        Returns:
            Dictionary with chunker configuration
        """
        return {
            "type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
