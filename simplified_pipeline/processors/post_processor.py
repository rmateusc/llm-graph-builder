"""
Post-processing module for graph enhancement
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class GraphPostProcessor:
    """Post-process and enhance the constructed graph"""

    def __init__(self, driver, database: str = "neo4j"):
        """
        Initialize post-processor

        Args:
            driver: Neo4j driver instance
            database: Database name
        """
        self.driver = driver
        self.database = database

    def merge_duplicate_nodes(self, similarity_threshold: float = 0.95):
        """
        Merge nodes that are likely duplicates based on similarity

        Args:
            similarity_threshold: Threshold for considering nodes as duplicates
        """
        with self.driver.session(database=self.database) as session:
            # Find and merge nodes with similar names
            query = """
            MATCH (n1), (n2)
            WHERE id(n1) < id(n2)
            AND labels(n1) = labels(n2)
            AND apoc.text.jaroWinklerDistance(n1.id, n2.id) > $threshold
            WITH n1, n2
            CALL apoc.refactor.mergeNodes([n1, n2], {
                properties: 'combine',
                mergeRels: true
            })
            YIELD node
            RETURN count(node) as merged_count
            """

            try:
                result = session.run(query, threshold=similarity_threshold)
                merged = result.single()['merged_count']
                logger.info(f"Merged {merged} duplicate nodes")
            except Exception as e:
                logger.warning(f"Could not merge duplicate nodes (APOC may not be installed): {e}")

    def create_communities(self):
        """
        Detect and create community structures in the graph
        """
        with self.driver.session(database=self.database) as session:
            try:
                # Use Louvain algorithm for community detection
                query = """
                CALL gds.graph.project(
                    'communities',
                    '*',
                    '*'
                )
                YIELD nodeCount, relationshipCount
                RETURN nodeCount, relationshipCount
                """
                session.run(query)

                # Run Louvain community detection
                query = """
                CALL gds.louvain.write('communities', {
                    writeProperty: 'community'
                })
                YIELD communityCount, modularity
                RETURN communityCount, modularity
                """
                result = session.run(query)
                stats = result.single()
                logger.info(f"Created {stats['communityCount']} communities with modularity {stats['modularity']}")

                # Clean up projection
                session.run("CALL gds.graph.drop('communities')")

            except Exception as e:
                logger.warning(f"Could not create communities (GDS may not be installed): {e}")

    def add_embeddings(self, embedding_model: Optional[Embeddings] = None):
        """
        Add embeddings to nodes for vector similarity search

        Args:
            embedding_model: Embeddings model to use
        """
        if not embedding_model:
            logger.warning("No embedding model provided, skipping embeddings")
            return

        with self.driver.session(database=self.database) as session:
            # Get all nodes with text content
            query = """
            MATCH (n)
            WHERE n.description IS NOT NULL
            RETURN id(n) as node_id, n.description as text
            """
            result = session.run(query)

            batch_size = 100
            batch = []

            for record in result:
                batch.append((record['node_id'], record['text']))

                if len(batch) >= batch_size:
                    self._process_embedding_batch(session, batch, embedding_model)
                    batch = []

            # Process remaining batch
            if batch:
                self._process_embedding_batch(session, batch, embedding_model)

            logger.info("Added embeddings to nodes")

    def _process_embedding_batch(self, session, batch, embedding_model):
        """Process a batch of nodes for embedding"""
        texts = [text for _, text in batch]
        embeddings = embedding_model.embed_documents(texts)

        for (node_id, _), embedding in zip(batch, embeddings):
            query = """
            MATCH (n)
            WHERE id(n) = $node_id
            SET n.embedding = $embedding
            """
            session.run(query, node_id=node_id, embedding=embedding)

    def create_vector_index(self, dimension: int = 1536):
        """
        Create vector index for similarity search

        Args:
            dimension: Dimension of the embeddings
        """
        with self.driver.session(database=self.database) as session:
            try:
                query = """
                CREATE VECTOR INDEX IF NOT EXISTS node_embeddings
                FOR (n:Node)
                ON (n.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: $dimension,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
                session.run(query, dimension=dimension)
                logger.info("Created vector index for embeddings")
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")

    def add_metadata_properties(self):
        """Add useful metadata properties to nodes and relationships"""
        with self.driver.session(database=self.database) as session:
            # Add creation timestamp
            query = """
            MATCH (n)
            WHERE n.created_at IS NULL
            SET n.created_at = datetime()
            """
            session.run(query)

            # Add degree centrality
            query = """
            MATCH (n)
            WITH n, size((n)--()) as degree
            SET n.degree = degree
            """
            session.run(query)

            logger.info("Added metadata properties to graph")

    def clean_graph(self):
        """Clean up the graph by removing isolated nodes and fixing issues"""
        with self.driver.session(database=self.database) as session:
            # Remove isolated nodes (no relationships)
            query = """
            MATCH (n)
            WHERE NOT (n)--()
            DELETE n
            RETURN count(n) as deleted_count
            """
            result = session.run(query)
            deleted = result.single()['deleted_count']
            logger.info(f"Removed {deleted} isolated nodes")

            # Remove self-loops
            query = """
            MATCH (n)-[r]->(n)
            DELETE r
            RETURN count(r) as deleted_count
            """
            result = session.run(query)
            deleted = result.single()['deleted_count']
            logger.info(f"Removed {deleted} self-loop relationships")

    def run_all_enhancements(self, embedding_model: Optional[Embeddings] = None):
        """
        Run all post-processing enhancements

        Args:
            embedding_model: Optional embedding model for vector embeddings
        """
        logger.info("Starting post-processing enhancements")

        self.clean_graph()
        self.merge_duplicate_nodes()
        self.add_metadata_properties()

        if embedding_model:
            self.add_embeddings(embedding_model)
            self.create_vector_index()

        self.create_communities()

        logger.info("Completed post-processing enhancements")