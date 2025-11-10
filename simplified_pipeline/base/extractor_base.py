"""
Abstract base class for entity and relationship extractors.

Defines the interface for all extraction strategies (LLM-based, rule-based, hybrid, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.schema import Document


class ExtractorBase(ABC):
    """Abstract base class for entity and relationship extraction"""

    def __init__(
        self,
        allowed_nodes: List[str] = None,
        allowed_relationships: List[str] = None,
        node_properties: List[str] = None,
        relationship_properties: List[str] = None
    ):
        """
        Initialize extractor with schema constraints

        Args:
            allowed_nodes: List of allowed entity types (None = extract all)
            allowed_relationships: List of allowed relationship types (None = extract all)
            node_properties: Properties to extract for nodes
            relationship_properties: Properties to extract for relationships
        """
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.node_properties = node_properties or ["description"]
        self.relationship_properties = relationship_properties or ["description"]

    @abstractmethod
    def extract_from_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract entities and relationships from document chunks

        Args:
            chunks: List of document chunks

        Returns:
            List of graph documents containing nodes and relationships
        """
        pass

    @abstractmethod
    def combine_graph_documents(self, graph_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple graph documents into a single unified graph

        Args:
            graph_documents: List of graph documents to combine

        Returns:
            Combined graph document with deduplicated nodes and relationships
        """
        pass

    def get_extractor_info(self) -> dict:
        """
        Get information about this extractor for experiment tracking

        Returns:
            Dictionary with extractor configuration
        """
        return {
            "type": self.__class__.__name__,
            "allowed_nodes": self.allowed_nodes,
            "allowed_relationships": self.allowed_relationships,
            "node_properties": self.node_properties,
            "relationship_properties": self.relationship_properties
        }
