from typing import List, Dict, Any, Tuple, Optional
from llm_util import LLMInterface
from neo4j_provider import Neo4jContextGraph
from retriever import Neo4jRetriever

class ToG3Constructor:
    """Agent that builds/modifies the heterogeneous graph on the fly."""
    def __init__(self, provider: Neo4jContextGraph, retriever: Neo4jRetriever):
        self.provider = provider
        self.retriever = retriever

    def evolve_subgraph(self, sub_query: str, current_nodes: List[str]) -> List[Dict[str, Any]]:
        """
        Dynamically expands the graph based on a sub-query.
        In a real ToG-3 system, this might involve extracting NEW triples from chunks.
        """
        print(f"Constructor: Evolving subgraph for query '{sub_query}'...")
        # For the prototype, we fetch k-hop neighbors for any new entities found in sub_query
        new_heads = self.retriever.retrieve_entities_by_label(sub_query)
        expanded_context = []
        for eid in new_heads:
            expanded_context.extend(self.retriever.get_k_hop_neighborhood(eid))
        return expanded_context

class ToG3Reflector:
    """Agent that assesses context sufficiency and manages query evolution."""
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def reflect(self, query: str, context: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Evaluates gathered context and suggests a sub-query for evolution.
        """
        # (Placeholder for real LLM logic using prompts like Appendix A.2)
        # We simplify for the iterative loop logic
        if not context:
            return False, f"Tell me more about the entities related to '{query}'"
        
        # Heuristic: if we have facts but they don't mention the query's core terms, evolve
        has_direct_match = any(query.lower() in str(c).lower() for c in context)
        if not has_direct_match and len(context) < 3:
            return False, f"Expand search for '{query}' by looking for indirect connections."
        
        return True, None

class ToG3Responser:
    """Agent responsible for final answer synthesis (Stage 3)."""
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def generate_final_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Synthesizes the final answer from all gathered context."""
        # Instruction: "Output all possible answers you can find IN THE MATERIALS"
        context_str = "\n".join([f"- {c['tr']['relation']} -> {c['tail']['label']}" for c in context])
        prompt = f"Context-aware Reasoning:\nQuery: {query}\n\nMaterials:\n{context_str}\n\nFinal Answer:"
        return self.llm.generate(prompt)
