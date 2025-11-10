"""
Main pipeline orchestrator for PDF to Graph conversion
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from extractors.pdf_processor import PDFProcessor
from processors.text_chunker import TextChunker
from extractors.entity_extractor import EntityRelationshipExtractor
from graph.graph_builder import Neo4jGraphBuilder
from processors.post_processor import GraphPostProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    # Neo4j settings
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"

    # Chunking settings
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Entity extraction settings
    allowed_nodes: List[str] = None
    allowed_relationships: List[str] = None
    node_properties: List[str] = None
    relationship_properties: List[str] = None

    # Post-processing settings
    enable_embeddings: bool = False
    enable_communities: bool = True
    merge_duplicates: bool = True
    similarity_threshold: float = 0.95


class PDFToGraphPipeline:
    """Main pipeline for converting PDFs to knowledge graphs"""

    def __init__(self, config: PipelineConfig, llm, embedding_model=None):
        """
        Initialize the pipeline

        Args:
            config: Pipeline configuration
            llm: Language model for entity extraction
            embedding_model: Optional embedding model for vector embeddings
        """
        self.config = config
        self.llm = llm
        self.embedding_model = embedding_model

        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.entity_extractor = EntityRelationshipExtractor(
            llm=llm,
            allowed_nodes=config.allowed_nodes,
            allowed_relationships=config.allowed_relationships,
            node_properties=config.node_properties,
            relationship_properties=config.relationship_properties
        )
        self.graph_builder = Neo4jGraphBuilder(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database
        )

    def process_pdf(self, pdf_path: str, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Process a PDF file and build a knowledge graph

        Args:
            pdf_path: Path to the PDF file
            clear_existing: Whether to clear existing graph data

        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info(f"Starting pipeline for: {pdf_path}")
        results = {}

        try:
            # Step 1: Load PDF
            logger.info("Step 1: Loading PDF")
            documents = self.pdf_processor.load_pdf(pdf_path)
            results['pages_loaded'] = len(documents)
            logger.info(f"Loaded {len(documents)} pages")

            # Step 2: Chunk documents
            logger.info("Step 2: Chunking documents")
            chunks = self.text_chunker.chunk_documents(documents)
            results['chunks_created'] = len(chunks)
            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Extract entities and relationships
            logger.info("Step 3: Extracting entities and relationships")
            graph_documents = self.entity_extractor.extract_from_chunks(chunks)
            results['chunks_processed'] = len(graph_documents)

            # Step 4: Combine graph documents
            logger.info("Step 4: Combining graph documents")
            combined_graph = self.entity_extractor.combine_graph_documents(graph_documents)
            results['total_nodes'] = len(combined_graph['nodes'])
            results['total_relationships'] = len(combined_graph['relationships'])

            # Step 5: Connect to Neo4j
            logger.info("Step 5: Connecting to Neo4j")
            self.graph_builder.connect()

            # Clear existing graph if requested
            if clear_existing:
                logger.info("Clearing existing graph")
                self.graph_builder.clear_graph()

            # Step 6: Build graph in Neo4j
            logger.info("Step 6: Building graph in Neo4j")
            self.graph_builder.build_graph(combined_graph)

            # Step 7: Post-processing
            logger.info("Step 7: Running post-processing")
            self._run_post_processing()

            # Get final statistics
            stats = self.graph_builder.get_graph_stats()
            results['final_node_count'] = stats['node_count']
            results['final_relationship_count'] = stats['relationship_count']

            logger.info("Pipeline completed successfully")
            results['status'] = 'success'

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise

        finally:
            # Close connection
            if hasattr(self, 'graph_builder'):
                self.graph_builder.close()

        return results

    def _run_post_processing(self):
        """Run post-processing enhancements"""
        post_processor = GraphPostProcessor(
            driver=self.graph_builder.driver,
            database=self.config.neo4j_database
        )

        if self.config.merge_duplicates:
            logger.info("Merging duplicate nodes")
            post_processor.merge_duplicate_nodes(self.config.similarity_threshold)

        logger.info("Adding metadata properties")
        post_processor.add_metadata_properties()

        if self.config.enable_embeddings and self.embedding_model:
            logger.info("Adding embeddings")
            post_processor.add_embeddings(self.embedding_model)
            post_processor.create_vector_index()

        if self.config.enable_communities:
            logger.info("Creating communities")
            post_processor.create_communities()

        logger.info("Cleaning graph")
        post_processor.clean_graph()

    def process_multiple_pdfs(self, pdf_paths: List[str], clear_existing: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files

        Args:
            pdf_paths: List of paths to PDF files
            clear_existing: Whether to clear existing graph before processing

        Returns:
            List of results for each PDF
        """
        results = []

        # Clear once if requested
        if clear_existing:
            self.graph_builder.connect()
            self.graph_builder.clear_graph()
            self.graph_builder.close()

        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"Processing PDF {i+1}/{len(pdf_paths)}: {pdf_path}")
            # Don't clear for subsequent PDFs
            result = self.process_pdf(pdf_path, clear_existing=False)
            results.append(result)

        return results