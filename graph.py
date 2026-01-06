from typing import List, Dict, Optional, Any
from models import ContextNode, ContextEdge, EntityContext, RelationContext

class ContextGraph:
    """An in-memory graph storage for entities and relations with context."""
    
    def __init__(self):
        self.nodes: Dict[str, ContextNode] = {}
        # adjacency list: head_id -> [ContextEdge]
        self.edges: Dict[str, List[ContextEdge]] = {}

    def add_node(self, node: ContextNode):
        """Adds a node to the graph."""
        self.nodes[node.entity_id] = node
        if node.entity_id not in self.edges:
            self.edges[node.entity_id] = []

    def add_edge(self, edge: ContextEdge):
        """Adds a directed edge between two nodes."""
        if edge.head not in self.nodes or edge.tail not in self.nodes:
            raise ValueError("Both head and tail nodes must exist in the graph.")
        self.edges[edge.head].append(edge)

    def get_node(self, entity_id: str) -> Optional[ContextNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(entity_id)

    def get_outgoing_edges(self, entity_id: str) -> List[ContextEdge]:
        """Retrieves all outgoing edges from a node."""
        return self.edges.get(entity_id, [])

    def search_by_label(self, label: str) -> List[ContextNode]:
        """Simple label-based search for entities."""
        return [node for node in self.nodes.values() if label.lower() in node.label.lower()]

    def get_triples_with_context(self, entity_id: str) -> List[Dict[str, Any]]:
        """Returns triples (h, r, t) with their full context for reasoning."""
        results = []
        head_node = self.nodes.get(entity_id)
        if not head_node:
            return []
            
        for edge in self.get_outgoing_edges(entity_id):
            tail_node = self.nodes.get(edge.tail)
            results.append({
                "head": head_node,
                "relation": edge.relation,
                "tail": tail_node,
                "context": edge.context
            })
        return results
