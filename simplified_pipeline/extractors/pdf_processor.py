"""
Simplified PDF processing module
Extracts text content from PDF files using PyMuPDF
"""
import logging
from pathlib import Path
from typing import List, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF files and extract text content"""

    def __init__(self):
        self.supported_extensions = ['.pdf']

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF file and extract text content

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects containing page content and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"File {file_path} is not a PDF")

        logger.info(f"Loading PDF: {file_path}")

        try:
            loader = PyMuPDFLoader(str(file_path))
            pages = loader.load()

            # Add page numbers to metadata if not present
            for i, page in enumerate(pages):
                if 'page' not in page.metadata:
                    page.metadata['page'] = i + 1
                page.metadata['source'] = str(file_path)
                page.metadata['filename'] = file_path.name

            logger.info(f"Successfully loaded {len(pages)} pages from {file_path}")
            return pages

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise

    def extract_text(self, documents: List[Document]) -> str:
        """
        Extract all text from documents

        Args:
            documents: List of Document objects

        Returns:
            Combined text from all documents
        """
        return "\n\n".join([doc.page_content for doc in documents])