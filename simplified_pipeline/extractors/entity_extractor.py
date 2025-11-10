"""
Entity and relationship extraction using LLMs
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.language_models.base import BaseLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity"""
    id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]


class EntityRelationshipExtractor:
    """Extract entities and relationships from text using LLMs"""

    def __init__(self,
                 llm: BaseLanguageModel,
                 allowed_nodes: List[str] = None,
                 allowed_relationships: List[str] = None,
                 node_properties: List[str] = None,
                 relationship_properties: List[str] = None):
        """
        Initialize the extractor

        Args:
            llm: Language model for extraction
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types
            node_properties: List of node properties to extract
            relationship_properties: List of relationship properties to extract
        """
        self.llm = llm
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.node_properties = node_properties or ["description"]
        self.relationship_properties = relationship_properties or ["description"]

        # Initialize LLM Graph Transformer
        self.graph_transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            strict_mode=False
        )

    def extract_from_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract entities and relationships from document chunks

        Args:
            chunks: List of document chunks

        Returns:
            List of graph documents containing nodes and relationships
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks")

        graph_documents = []

        for i, chunk in enumerate(chunks):
            try:
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")

                # Convert chunk to graph document
                graph_docs = self.graph_transformer.convert_to_graph_documents([chunk])

                if graph_docs:
                    graph_doc = graph_docs[0]

                    # Add chunk reference to nodes and relationships
                    for node in graph_doc.nodes:
                        node.properties['chunk_id'] = chunk.metadata.get('chunk_id', i)
                        node.properties['source'] = chunk.metadata.get('source', '')

                    for rel in graph_doc.relationships:
                        rel.properties['chunk_id'] = chunk.metadata.get('chunk_id', i)
                        rel.properties['source'] = chunk.metadata.get('source', '')

                    graph_documents.append({
                        'chunk_id': chunk.metadata.get('chunk_id', i),
                        'nodes': graph_doc.nodes,
                        'relationships': graph_doc.relationships,
                        'source': chunk.metadata.get('source', ''),
                        'page': chunk.metadata.get('page', None)
                    })

            except Exception as e:
                logger.warning(f"Failed to extract from chunk {i}: {e}")
                continue

        logger.info(f"Extracted graph documents from {len(graph_documents)} chunks")
        return graph_documents

    def combine_graph_documents(self, graph_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple graph documents into a single graph

        Args:
            graph_documents: List of graph documents

        Returns:
            Combined graph with deduplicated nodes and relationships
        """
        combined_nodes = {}
        combined_relationships = []

        for doc in graph_documents:
            # Add nodes (deduplicate by id)
            for node in doc.get('nodes', []):
                node_id = node.id
                if node_id not in combined_nodes:
                    combined_nodes[node_id] = node
                else:
                    # Merge properties if node already exists
                    existing_node = combined_nodes[node_id]
                    for key, value in node.properties.items():
                        if key not in existing_node.properties:
                            existing_node.properties[key] = value

            # Add relationships
            for rel in doc.get('relationships', []):
                combined_relationships.append(rel)

        logger.info(f"Combined into {len(combined_nodes)} nodes and {len(combined_relationships)} relationships")

        return {
            'nodes': list(combined_nodes.values()),
            'relationships': combined_relationships
        }