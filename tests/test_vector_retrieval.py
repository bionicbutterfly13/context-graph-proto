import pytest
from neo4j_provider import Neo4jContextGraph
from retriever import Neo4jRetriever
from models import ChunkNode

@pytest.fixture
def provider():
    p = Neo4jContextGraph("bolt://localhost:7687", "neo4j", "not-a-real-pass")
    try:
        p.query("MATCH (n) DETACH DELETE n")
        p.initialize_schema()
        yield p
    finally:
        p.close()

def test_chunk_retrieval(provider):
    """Verifies that chunks can be retrieved via the search_chunks dual pathway."""
    c1 = ChunkNode("C1", "The law of the photoelectric effect was discovered by Einstein.")
    provider.add_chunk(c1)
    
    retriever = Neo4jRetriever(provider)
    results = retriever.search_chunks("photoelectric")
    
    assert len(results) == 1
    assert "photoelectric" in results[0]['c']['content']
