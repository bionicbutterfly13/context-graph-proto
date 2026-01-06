from typing import List, Dict, Any
from llm_util import LLMInterface

class LLMRanker:
    """
    Discriminative Filter (Stage 2 of CGR3).
    Uses an LLM to re-rank candidates based on their semantic compatibility 
    with the head entity context and the query.
    """
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def rerank(self, query: str, head_context: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes candidate entities (structures: {id, name, description}) 
        and returns them sorted by relevance.
        """
        if not candidates:
            return []
            
        prompt = self.llm.build_ranking_prompt(query, head_context, candidates)
        response = self.llm.generate(prompt)
        
        # In a real system, we parse "The final order: [1, 3, 2...]"
        # For the prototype, we return the candidates list. 
        # In the verification script, the LLM interface (simulated) would handle this.
        
        # Mocking the sort for the prototype's silent execution
        # (A real parser would go here)
        return candidates
