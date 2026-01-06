import pytest
from models import ContextNode, ContextEdge, EntityContext, RelationContext
from graph import ContextGraph
from retriever import MockRetriever
from ranker import MockRanker
from reasoner import ContextReasoner

@pytest.fixture
def sample_graph():
    cg = ContextGraph()
    n1 = ContextNode("E1", "Entity1")
    n2 = ContextNode("E2", "Entity2")
    cg.add_node(n1)
    cg.add_node(n2)
    edge = ContextEdge("E1", "relates_to", "E2", RelationContext(temporal={"time": "now"}))
    cg.add_edge(edge)
    return cg

def test_graph_add_node(sample_graph):
    assert len(sample_graph.nodes) == 2
    assert sample_graph.get_node("E1").label == "Entity1"

def test_graph_add_edge(sample_graph):
    edges = sample_graph.get_outgoing_edges("E1")
    assert len(edges) == 1
    assert edges[0].relation == "relates_to"
    assert edges[0].context.temporal == {"time": "now"}

def test_retriever_entities(sample_graph):
    retriever = MockRetriever(sample_graph)
    entities = retriever.retrieve_topic_entities("Tell me about Entity1")
    assert len(entities) == 1
    assert entities[0].entity_id == "E1"

def test_reasoner_one_hop(sample_graph):
    retriever = MockRetriever(sample_graph)
    ranker = MockRanker()
    reasoner = ContextReasoner(retriever, ranker, max_hops=1)
    results = reasoner.reason("Tell me about Entity1")
    assert len(results) == 1
    assert results[0]['data']['relation'] == "relates_to"
    assert results[0]['hop'] == 1
