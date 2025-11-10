"""
Text chunking module for splitting documents into processable chunks
"""
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextChunker:
    """Split text into chunks for processing"""

    def __init__(self,
                 chunk_size: int = 2000,
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        """
        Initialize text chunker

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators for splitting (default: common text separators)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents")

        chunks = self.splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text string into chunks

        Args:
            text: Text string to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunked Document objects
        """
        metadata = metadata or {}
        chunks = self.splitter.split_text(text)

        documents = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk_text)
            })
            documents.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))

        logger.info(f"Created {len(documents)} chunks from text")
        return documents