from typing import List, Dict, Any
from models import ContextNode, ContextEdge, EntityContext, RelationContext, ChunkNode, CommunityNode
from neo4j_provider import Neo4jContextGraph

class Neo4jRetriever:
    """Retriever that fetches multi-level context from Neo4j."""
    
    def __init__(self, provider: Neo4jContextGraph):
        self.provider = provider

    def retrieve_entities_by_label(self, label_substring: str) -> List[str]:
        """Search Neo4j for entities matching a label."""
        cypher = "MATCH (e:Entity) WHERE e.label CONTAINS $label RETURN e.id as id"
        results = self.provider.query(cypher, {"label": label_substring})
        return [res['id'] for res in results]

    def fetch_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """Fetches the EC (Entity Context) including attributes and metadata."""
        cypher = "MATCH (e:Entity {id: $id}) RETURN e"
        result = self.provider.query(cypher, {"id": entity_id})
        return result[0]['e'] if result else {}

    def fetch_community_context(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetches community summaries for a set of entities."""
        cypher = """
        MATCH (e:Entity)-[:PART_OF]->(m:Community)
        WHERE e.id IN $ids
        RETURN DISTINCT m.id as id, m.label as label, m.summary as summary
        """
        return self.provider.query(cypher, {"ids": entity_ids})

    def fetch_chunk_context(self, triplet_id: str) -> List[Dict[str, Any]]:
        """Fetches raw text chunks supporting a specific triplet."""
        cypher = """
        MATCH (tr:Triplet {id: $tid})-[:EVIDENCE_IN]->(c:Chunk)
        RETURN c.id as id, c.content as content, c.metadata as metadata
        """
        return self.provider.query(cypher, {"tid": triplet_id})

    def fetch_fewshot_triples(self, relation: str, limit: int = 3) -> str:
        """Fetches sample triples for a relation to serve as Type-Aware context."""
        cypher = """
        MATCH (h:Entity)-[:HAS_SUBJECT_OF]->(tr:Triplet {relation: $rel})-[:HAS_OBJECT_OF]->(t:Entity)
        RETURN h.label as head, tr.relation as relation, t.label as tail
        LIMIT $limit
        """
        results = self.provider.query(cypher, {"rel": relation, "limit": limit})
        return "\n".join([f"({r['head']}, {r['relation']}, {r['tail']})" for r in results])

    def fetch_reasoning_paths(self, head_id: str, tail_id: str, max_hops: int = 3) -> str:
        """Official CATS Subgraph Reasoning: search for paths between head and tail."""
        cypher = """
        MATCH p = (h:Entity {id: $head_id})-[*1..3]-(t:Entity {id: $tail_id})
        RETURN p
        LIMIT 5
        """
        # In a real system, we format the path as a sequence of triples
        results = self.provider.query(cypher, {"head_id": head_id, "tail_id": tail_id})
        return f"Found {len(results)} paths between entities."

    def search_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Dual Pathway: Textual Context Retrieval."""
        return self.provider.search_chunks_vector(query)

    def get_k_hop_neighborhood(self, entity_id: str, k: int = 1) -> List[Dict[str, Any]]:
        """Pathway A: Structural Retrieval (Simplified KGE proxy)."""
        return self.provider.get_neighbors(entity_id)
