from typing import List, Dict, Any, Tuple, Optional
from neo4j_provider import Neo4jContextGraph
from retriever import Neo4jRetriever
from ranker import LLMRanker
from llm_util import LLMInterface

class MACERConstructor:
    """Agent that builds a task-specific sub-graph context."""
    def __init__(self, provider: Neo4jContextGraph, retriever: Neo4jRetriever):
        self.provider = provider
        self.retriever = retriever

    def expand(self, entity_id: str) -> List[Dict[str, Any]]:
        """Pathway A: Structural Retrieval of neighborhood."""
        return self.retriever.get_k_hop_neighborhood(entity_id)

class MACERReflector:
    """Agent that evaluates context sufficiency using LLM (Sufficiency Gate)."""
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def evaluate(self, query: str, head_context: str, top_candidate: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Stage 3: Context-Aware Reasoning (Appendix A.2).
        Returns (is_sufficient, response).
        """
        # Format candidate for prompt
        # cand format: {tr: {...}, tail: {...}, chunks: [...]}
        tr = top_candidate.get('tr', {})
        tail = top_candidate.get('tail', {})
        
        cand_info = {
            "name": tail.get('label', 'Unknown'),
            "description": tail.get('metadata', 'No description available'),
            "evidence": " ".join([c['content'] for c in top_candidate.get('chunks', [])])
        }
        
        prompt = self.llm.build_reasoning_prompt(query, head_context, cand_info)
        response = self.llm.generate(prompt)
        
        # Check if LLM indicates sufficiency
        # In a real system, we look for 'The possible answers:'
        if "possible answers:" in response.lower() and "insufficient information" not in response.lower():
            return True, response
        
        return False, response

class MACERReasoner:
    """CGR3 Paradigm: Retrieve -> Rank -> Reason."""
    
    def __init__(self, provider: Neo4jContextGraph, llm: LLMInterface, max_iterations: int = 3):
        self.provider = provider
        self.retriever = Neo4jRetriever(provider)
        self.llm = llm
        self.constructor = MACERConstructor(provider, self.retriever)
        self.ranker = LLMRanker(llm)
        self.reflector = MACERReflector(llm)
        self.max_iterations = max_iterations

    def reason(self, query: str) -> Dict[str, Any]:
        """The CGR3 / MACER reasoning loop."""
        # 1. Extract Head Entities (Dual Pathway Stage 1)
        # Simplified: find entities mentioned in query
        head_ids = self.retriever.retrieve_entities_by_label(query)
        
        all_found_context = []
        final_answer = None
        
        for head_id in head_ids:
            # Get Head Context (EC)
            head_node = self.retriever.fetch_entity_context(head_id)
            head_context_text = head_node.get('metadata', '')
            
            # Stage 1: Retrieval (Structural Neighborhood)
            candidates = self.constructor.expand(head_id)
            
            # Stage 2: Context-Aware Ranking (Discriminative Filter)
            # Format candidates for the ranker
            formatted_candidates = []
            for cand in candidates:
                tail = cand.get('tail', {})
                formatted_candidates.append({
                    "id": tail.get('id'),
                    "name": tail.get('label'),
                    "description": tail.get('metadata', 'No context')
                })
            
            # Re-rank using LLM
            ranked_indices = self.ranker.rerank(query, head_context_text, formatted_candidates)
            
            # Path Pruning: Only proceed with top-N candidates
            # For this prototype, we'll take the top 1
            if candidates:
                top_candidate = candidates[0] # Assuming ranker sorted them
                
                # Stage 3: Reasoning (Sufficiency Gate)
                is_sufficient, response = self.reflector.evaluate(query, head_context_text, top_candidate)
                
                if is_sufficient:
                    final_answer = response
                    all_found_context.append(top_candidate)
                    break
                else:
                    # In ToG-3, this would trigger Dual-Evolution (evolution_query)
                    # For CGR3, it might just report insufficiency or move to next candidate
                    print(f"Reflector: Context for {head_id} insufficient. Proceeding...")
                    all_found_context.append(top_candidate)

        return {
            "query": query,
            "answer": final_answer,
            "final_context": all_found_context
        }
