#!/usr/bin/env python3
"""
Script to clear all data from Neo4j Aura cloud database
"""

import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jCleaner:
    """Class to handle clearing Neo4j database"""
    
    def __init__(self):
        """Initialize Neo4j connection"""
        # Neo4j configuration (cloud instance from week_3)
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j+s://9d46ce62.databases.neo4j.io")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "55f8fZo2BMDYi4LsicP5xgkUZN2GlM_aT8zaMKE8920")
        
        # Initialize driver
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            logger.info(f"✅ Connected to Neo4j at {self.neo4j_uri}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def get_database_stats(self):
        """Get current database statistics"""
        try:
            with self.driver.session() as session:
                # Get node counts by label
                node_counts = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(n) as count
                    ORDER BY count DESC
                """)
                
                # Get relationship counts by type
                rel_counts = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                """)
                
                # Get total counts
                total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                
                logger.info(f"📊 Database Statistics:")
                logger.info(f"   Total Nodes: {total_nodes}")
                logger.info(f"   Total Relationships: {total_rels}")
                
                logger.info(f"   Nodes by Label:")
                for record in node_counts:
                    labels = ", ".join(record["labels"]) if record["labels"] else "no-label"
                    logger.info(f"     {labels}: {record['count']}")
                
                logger.info(f"   Relationships by Type:")
                for record in rel_counts:
                    logger.info(f"     {record['type']}: {record['count']}")
                
                return {
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "node_counts": {",".join(record["labels"]) if record["labels"] else "no-label": record["count"] for record in node_counts},
                    "relationship_counts": {record["type"]: record["count"] for record in rel_counts}
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return None
    
    def clear_all_data(self):
        """Clear all data from the database"""
        try:
            with self.driver.session() as session:
                logger.info("🗑️  Starting database cleanup...")
                
                # Delete all relationships first
                result = session.run("MATCH ()-[r]->() DELETE r")
                logger.info(f"✅ Deleted {result.consume().counters.relationships_deleted} relationships")
                
                # Delete all nodes
                result = session.run("MATCH (n) DELETE n")
                logger.info(f"✅ Deleted {result.consume().counters.nodes_deleted} nodes")
                
                # Clear any constraints and indexes (optional - will be recreated)
                try:
                    # Drop constraints
                    constraints = session.run("SHOW CONSTRAINTS")
                    for record in constraints:
                        constraint_name = record["name"]
                        session.run(f"DROP CONSTRAINT {constraint_name}")
                        logger.info(f"✅ Dropped constraint: {constraint_name}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not drop constraints: {e}")
                
                try:
                    # Drop indexes (except built-in ones)
                    indexes = session.run("SHOW INDEXES")
                    for record in indexes:
                        index_name = record["name"]
                        if not index_name.startswith("neo4j_"):  # Skip built-in indexes
                            session.run(f"DROP INDEX {index_name}")
                            logger.info(f"✅ Dropped index: {index_name}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not drop indexes: {e}")
                
                logger.info("🎉 Database cleanup completed successfully!")
                
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")

def main():
    """Main function to clear the database"""
    cleaner = None
    try:
        # Create cleaner instance
        cleaner = Neo4jCleaner()
        
        # Show current stats
        logger.info("📈 Current database statistics:")
        stats = cleaner.get_database_stats()
        
        if stats and (stats["total_nodes"] > 0 or stats["total_relationships"] > 0):
            # Ask for confirmation
            response = input("\n⚠️  This will DELETE ALL DATA from the Neo4j database. Are you sure? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                cleaner.clear_all_data()
                
                # Show final stats
                logger.info("\n📈 Final database statistics:")
                final_stats = cleaner.get_database_stats()
                
                if final_stats and final_stats["total_nodes"] == 0 and final_stats["total_relationships"] == 0:
                    logger.info("✅ Database successfully cleared!")
                else:
                    logger.warning("⚠️  Database may not be completely cleared")
            else:
                logger.info("❌ Database cleanup cancelled")
        else:
            logger.info("ℹ️  Database is already empty")
            
    except Exception as e:
        logger.error(f"❌ Error during database cleanup: {e}")
    finally:
        if cleaner:
            cleaner.close()

if __name__ == "__main__":
    main() 