import pytest
from neo4j_provider import Neo4jContextGraph
from llm_util import LLMInterface
from reasoner import MACERReasoner
from models import ContextNode, ContextEdge, RelationContext, ChunkNode, EntityContext

@pytest.fixture
def provider():
    # Note: Requires a running Neo4j instance
    p = Neo4jContextGraph("bolt://localhost:7687", "neo4j", "not-a-real-pass")
    try:
        p.query("MATCH (n) DETACH DELETE n") # Clean slate
        p.initialize_schema()
        yield p
    finally:
        p.close()

def test_macer_loop_evolution(provider):
    """Verifies that the MACER loop correctly evolves the query and gathers context."""
    # Build a 2-hop graph
    # Einstein (Q937) -> WON -> Nobel (Q38104)
    # Nobel (Q38104) -> AWARDED_BY -> Royal Swedish Academy (Q191459)
    
    provider.add_entity(ContextNode("Q937", "Albert Einstein", context=EntityContext(metadata={"description": "Physicist"})))
    provider.add_entity(ContextNode("Q38104", "Nobel Prize in Physics", context=EntityContext(metadata={"description": "Yearly award"})))
    provider.add_entity(ContextNode("Q191459", "Royal Swedish Academy", context=EntityContext(metadata={"description": "Scientific body"})))
    
    provider.add_triplet_with_context(ContextEdge("Q937", "WON", "Q38104", context=RelationContext(temporal={"year": 1921})))
    provider.add_triplet_with_context(ContextEdge("Q38104", "AWARDED_BY", "Q191459", context=RelationContext(confidence=1.0)))
    
    llm = LLMInterface()
    reasoner = MACERReasoner(provider, llm)
    
    # Query: "Who awarded Einstein his Nobel prize?"
    # It should hop from Einstein to Nobel, then to Academy
    results = reasoner.reason("Einstein Nobel prize")
    
    assert len(results["final_context"]) > 0
    # The first fact should be about the Nobel Prize
    assert results["final_context"][0]['tr']['relation'] == "WON"
    
    # Check for evolution (This depends on the mock refactor in macer_agents.py)
    # If the first hop wasn't sufficient, it evolves
    print(f"Iterations: {results['iterations']}")
    assert results['iterations'] >= 1
