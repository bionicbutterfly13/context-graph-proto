from typing import List, Dict, Any, Tuple
from llm_util import LLMInterface
from retriever import Neo4jRetriever

class LLMRanker:
    """
    Discriminative Filter (Stage 2 of CGR3) enhanced with CATS logic.
    Uses an LLM to re-rank candidates based on:
    1. Semantic compatibility (Head/Tail descriptions).
    2. Type-aware reasoning (Entity types vs Relation constraints).
    3. Subgraph reasoning (Neighboring facts and paths).
    """
    
    def __init__(self, llm: LLMInterface, retriever: Neo4jRetriever):
        self.llm = llm
        self.retriever = retriever

    def rerank(self, query: str, head_id: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes candidate entities and returns them sorted by relevance, 
        applying the CATS multi-context perspective.
        """
        if not candidates:
            return []
            
        head_node = self.retriever.fetch_entity_context(head_id)
        head_context = head_node.get('metadata', {}).get('description', '')
        
        # We assume the query can be parsed into a relation. 
        # Simplified: if "WON" is in query, we assume relation is WON.
        relation = "WON" if "WON" in query.upper() else "RELATED_TO"
        
        scored_candidates = []
        for cand in candidates:
            tail_id = cand['id']
            tail_name = cand['name']
            tail_desc = cand['description']
            
            # 1. Type-Aware Reasoning (TAR)
            fewshot = self.retriever.fetch_fewshot_triples(relation)
            test_triple = f"({head_id}, {relation}, {tail_name})"
            type_prompt = self.llm.build_type_reasoning_prompt(test_triple, fewshot)
            type_score = 1.0 if self.llm.generate(type_prompt) == "Y" else 0.5
            
            # 2. Subgraph Reasoning (SR)
            neighbor_triples = self.retriever.fetch_fewshot_triples(relation, limit=5) # Neighbors proxy
            reasoning_paths = self.retriever.fetch_reasoning_paths(head_id, tail_id)
            subgraph_prompt = self.llm.build_subgraph_reasoning_prompt(test_triple, neighbor_triples, reasoning_paths)
            subgraph_score = 1.0 if self.llm.generate(subgraph_prompt) == "Y" else 0.5
            
            # Aggregate Score (Simplified)
            total_score = type_score + subgraph_score
            scored_candidates.append((total_score, cand))
            
        # Sort by total score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in scored_candidates]
