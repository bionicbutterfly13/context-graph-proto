from typing import List, Dict, Any
from graph import ContextGraph
from models import ContextNode, ContextEdge, EntityContext, RelationContext

class MockRetriever:
    """Simulates the 'Retrieval' stage of CGR^3."""
    
    def __init__(self, graph: ContextGraph):
        self.graph = graph

    def retrieve_topic_entities(self, query: str) -> List[ContextNode]:
        """Search graph for entities mentioned in query."""
        # Simple simulation: look for matching substrings in the query
        found = []
        for node in self.graph.nodes.values():
            if node.label.lower() in query.lower():
                found.append(node)
        return found

    def fetch_triples_and_context(self, entity_id: str) -> List[Dict[str, Any]]:
        """Fetch relations and their context for a given entity."""
        return self.graph.get_triples_with_context(entity_id)

    def fetch_supporting_sentences(self, head_label: str, tail_label: str) -> List[str]:
        """Simulates Wikipedia context retrieval using semantic similarity."""
        # In a real system, this would use embeddings and an external corpus.
        return [
            f"{head_label} and {tail_label} are often cited together in historical records.",
            f"The relationship between {head_label} and {tail_label} was established during the early 20th century."
        ]
