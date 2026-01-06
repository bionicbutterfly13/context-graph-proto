from neo4j import GraphDatabase
import json
from typing import List, Dict, Any, Optional
from models import ContextNode, ContextEdge, EntityContext, RelationContext, ChunkNode, CommunityNode

class Neo4jContextGraph:
    """Enhanced Context Graph powered by Neo4j and ToG-3 concepts."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher, parameters=None):
        with self.driver.session() as session:
            return session.run(cypher, parameters).data()

    def initialize_schema(self):
        """Creates indexes and constraints."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Community) REQUIRE m.id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.label)",
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    def add_entity(self, node: ContextNode):
        cypher = """
        MERGE (e:Entity {id: $id})
        SET e.label = $label,
            e.attributes = $attributes,
            e.metadata = $metadata,
            e.external_links = $links
        """
        params = {
            "id": node.entity_id,
            "label": node.label,
            "attributes": json.dumps(node.context.attributes),
            "metadata": json.dumps(node.context.metadata),
            "links": node.context.external_links
        }
        self.query(cypher, params)

    def add_chunk(self, chunk: ChunkNode):
        cypher = """
        MERGE (c:Chunk {id: $id})
        SET c.content = $content,
            c.metadata = $metadata
        """
        params = {
            "id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": json.dumps(chunk.metadata)
        }
        self.query(cypher, params)

    def add_community(self, community: CommunityNode):
        cypher = """
        MERGE (m:Community {id: $id})
        SET m.label = $label,
            m.summary = $summary
        """
        self.query(cypher, {"id": community.community_id, "label": community.label, "summary": community.summary})
        
        # Link entities to community
        for entity_id in community.entities:
            self.query("""
            MATCH (e:Entity {id: $entity_id}), (m:Community {id: $community_id})
            MERGE (e)-[:PART_OF]->(m)
            """, {"entity_id": entity_id, "community_id": community.community_id})

    def add_triplet_with_context(self, edge: ContextEdge, chunk_ids: List[str] = None):
        """
        Adds a reified Triplet node with rich Relation Context (rc).
        Follows the (h, r, t, rc) quadruple structure.
        """
        triplet_id = f"{edge.head}_{edge.relation}_{edge.tail}"
        
        # Flatten Relation Context for Neo4j properties
        rc_props = {
            "id": triplet_id,
            "relation": edge.relation,
            "confidence": edge.context.confidence,
            "geographic": edge.context.geographic,
            "provenance": json.dumps(edge.context.provenance)
        }
        
        if edge.context.temporal:
            for k, v in edge.context.temporal.items():
                rc_props[f"temporal_{k}"] = v
        
        if edge.context.details:
            for k, v in edge.context.details.items():
                rc_props[f"prop_{k}"] = v

        cypher = """
        MATCH (h:Entity {id: $head_id})
        MATCH (t:Entity {id: $tail_id})
        MERGE (tr:Triplet {id: $tid})
        SET tr += $props
        MERGE (h)-[:HAS_SUBJECT_OF]->(tr)
        MERGE (tr)-[:HAS_OBJECT_OF]->(t)
        """
        self.query(cypher, {
            "head_id": edge.head,
            "tail_id": edge.tail,
            "tid": triplet_id,
            "props": rc_props
        })

        if chunk_ids:
            for cid in chunk_ids:
                self.query("""
                    MATCH (tr:Triplet {id: $tid}), (c:Chunk {id: $cid})
                    MERGE (tr)-[:EVIDENCE_IN]->(c)
                """, {"tid": triplet_id, "cid": cid})

    def get_relation_context(self, triplet_id: str) -> Dict[str, Any]:
        """Fetches the RC (Relation Context) for a quadruple."""
        cypher = "MATCH (tr:Triplet {id: $tid}) RETURN tr"
        result = self.query(cypher, {"tid": triplet_id})
        return result[0]['tr'] if result else {}

    def search_chunks_vector(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Stub for Neo4j Vector Index search.
        In a real system, this uses db.index.vector.queryNodes.
        """
        # Fallback to keyword search for prototype
        cypher = "MATCH (c:Chunk) WHERE c.content CONTAINS $query RETURN c LIMIT $limit"
        return self.query(cypher, {"query": query, "limit": limit})

    def get_neighbors(self, entity_id: str) -> List[Dict[str, Any]]:
        """Retrieves neighbors and their contexts (ToG-3 style traversal)."""
        cypher = """
        MATCH (e:Entity {id: $id})-[r1:HAS_SUBJECT_OF]->(tr:Triplet)-[r2:HAS_OBJECT_OF]->(tail:Entity)
        OPTIONAL MATCH (tr)-[:EVIDENCE_IN]->(c:Chunk)
        OPTIONAL MATCH (e)-[:PART_OF]->(m:Community)
        RETURN e, tr, tail, collect(c) as chunks, collect(m) as communities
        """
        return self.query(cypher, {"id": entity_id})
