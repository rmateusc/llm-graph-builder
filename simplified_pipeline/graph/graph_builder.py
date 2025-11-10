"""
Graph construction module for Neo4j
"""
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)


class Neo4jGraphBuilder:
    """Build and manage graph in Neo4j database"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: neo4j)
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None

    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def create_constraints(self, node_types: List[str]):
        """
        Create uniqueness constraints for node types

        Args:
            node_types: List of node types to create constraints for
        """
        with self.driver.session(database=self.database) as session:
            for node_type in node_types:
                try:
                    query = f"""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type})
                    REQUIRE n.id IS UNIQUE
                    """
                    session.run(query)
                    logger.info(f"Created constraint for {node_type}")
                except Neo4jError as e:
                    logger.warning(f"Could not create constraint for {node_type}: {e}")

    def create_indexes(self, node_types: List[str], properties: List[str]):
        """
        Create indexes for better query performance

        Args:
            node_types: List of node types
            properties: List of properties to index
        """
        with self.driver.session(database=self.database) as session:
            for node_type in node_types:
                for prop in properties:
                    try:
                        query = f"""
                        CREATE INDEX IF NOT EXISTS FOR (n:{node_type})
                        ON (n.{prop})
                        """
                        session.run(query)
                        logger.info(f"Created index on {node_type}.{prop}")
                    except Neo4jError as e:
                        logger.warning(f"Could not create index on {node_type}.{prop}: {e}")

    def create_nodes(self, nodes: List[Any]) -> int:
        """
        Create nodes in the graph

        Args:
            nodes: List of node objects

        Returns:
            Number of nodes created
        """
        with self.driver.session(database=self.database) as session:
            created_count = 0

            for node in nodes:
                try:
                    # Build properties string
                    props = {
                        'id': node.id,
                        **node.properties
                    }

                    # Create MERGE query
                    query = f"""
                    MERGE (n:{node.type} {{id: $id}})
                    SET n += $props
                    RETURN n
                    """

                    result = session.run(query, id=node.id, props=props)
                    if result.single():
                        created_count += 1

                except Exception as e:
                    logger.error(f"Failed to create node {node.id}: {e}")

            logger.info(f"Created/merged {created_count} nodes")
            return created_count

    def create_relationships(self, relationships: List[Any]) -> int:
        """
        Create relationships in the graph

        Args:
            relationships: List of relationship objects

        Returns:
            Number of relationships created
        """
        with self.driver.session(database=self.database) as session:
            created_count = 0

            for rel in relationships:
                try:
                    # Build properties
                    props = rel.properties or {}

                    # Create MERGE query
                    query = f"""
                    MATCH (source {{id: $source_id}})
                    MATCH (target {{id: $target_id}})
                    MERGE (source)-[r:{rel.type}]->(target)
                    SET r += $props
                    RETURN r
                    """

                    result = session.run(
                        query,
                        source_id=rel.source.id if hasattr(rel, 'source') else rel.source_id,
                        target_id=rel.target.id if hasattr(rel, 'target') else rel.target_id,
                        props=props
                    )
                    if result.single():
                        created_count += 1

                except Exception as e:
                    logger.error(f"Failed to create relationship: {e}")

            logger.info(f"Created {created_count} relationships")
            return created_count

    def build_graph(self, graph_data: Dict[str, Any]):
        """
        Build complete graph from extracted data

        Args:
            graph_data: Dictionary containing nodes and relationships
        """
        nodes = graph_data.get('nodes', [])
        relationships = graph_data.get('relationships', [])

        logger.info(f"Building graph with {len(nodes)} nodes and {len(relationships)} relationships")

        # Create nodes
        self.create_nodes(nodes)

        # Create relationships
        self.create_relationships(relationships)

        logger.info("Graph construction completed")

    def clear_graph(self):
        """Clear all nodes and relationships from the graph"""
        with self.driver.session(database=self.database) as session:
            try:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared graph database")
            except Neo4jError as e:
                logger.error(f"Failed to clear graph: {e}")
                raise

    def get_graph_stats(self) -> Dict[str, int]:
        """
        Get statistics about the graph

        Returns:
            Dictionary with node and relationship counts
        """
        with self.driver.session(database=self.database) as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']

            return {
                'node_count': node_count,
                'relationship_count': rel_count
            }