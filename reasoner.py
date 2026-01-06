from typing import List, Dict, Any, Tuple, Optional
from neo4j_provider import Neo4jContextGraph
from retriever import Neo4jRetriever
from ranker import LLMRanker
from llm_util import LLMInterface
from macer_agents import ToG3Constructor, ToG3Reflector, ToG3Responser

class MACERReasoner:
    """
    ToG-3 Orchestrator: Dual-Evolution of Query and Subgraph.
    Integrates CGR3 Retrieve-Rank-Reason with ToG-3 iteration.
    """
    
    def __init__(self, provider: Neo4jContextGraph, llm: LLMInterface, max_iterations: int = 3):
        self.provider = provider
        self.retriever = Neo4jRetriever(provider)
        self.llm = llm
        self.constructor = ToG3Constructor(provider, self.retriever)
        self.ranker = LLMRanker(llm, self.retriever)
        self.reflector = ToG3Reflector(llm)
        self.responser = ToG3Responser(llm)
        self.max_iterations = max_iterations

    def reason(self, query: str) -> Dict[str, Any]:
        """The official ToG-3 MACER reasoning loop."""
        current_query = query
        all_gathered_context = []
        iteration = 0
        final_answer = None
        
        while iteration < self.max_iterations:
            print(f"--- MACER Iteration {iteration+1} for Query: '{current_query}' ---")
            
            # 1. Retrieval (Dual Pathway)
            head_ids = self.retriever.retrieve_entities_by_label(current_query)
            if not head_ids:
                # Fallback to search chunks if no entities found
                relevant_chunks = self.retriever.search_chunks(current_query)
                # (Simplified: just log for prototype)
            
            iteration_context = []
            for head_id in head_ids:
                # Stage 1: Retrieval (Structural Neighborhood)
                candidates = self.constructor.evolve_subgraph(current_query, [head_id])
                
                # Format candidates for ranker (id, name, description)
                formatted_candidates = []
                for cand in candidates:
                    tail = cand.get('tail', {})
                    formatted_candidates.append({
                        "id": tail.get('id'),
                        "name": tail.get('label'),
                        "description": tail.get('metadata', 'No context')
                    })
                
                # Stage 2: CATS-Enhanced Ranking
                ranked_candidates = self.ranker.rerank(current_query, head_id, formatted_candidates)
                
                # Add top ranked candidates to total context
                if ranked_candidates:
                    # Map back to original detailed candidate objects
                    top_id = ranked_candidates[0]['id']
                    top_detailed = next((c for c in candidates if c['tail'].get('id') == top_id), None)
                    if top_detailed:
                        iteration_context.append(top_detailed)

            all_gathered_context.extend(iteration_context)
            
            # 2. Reflection (Sufficiency Gate & Evolution Query)
            is_sufficient, evolution_query = self.reflector.reflect(query, all_gathered_context)
            
            if is_sufficient:
                print("Reflector: Context is sufficient. Generating final answer.")
                final_answer = self.responser.generate_final_answer(query, all_gathered_context)
                break
            else:
                print(f"Reflector: Context insufficient. Evolving query to: '{evolution_query}'")
                current_query = evolution_query
                iteration += 1

        if not final_answer and all_gathered_context:
            final_answer = self.responser.generate_final_answer(query, all_gathered_context)

        return {
            "query": query,
            "answer": final_answer,
            "final_context": all_gathered_context,
            "iterations": iteration + 1
        }
