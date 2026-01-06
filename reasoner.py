from typing import List, Dict, Any, Optional
from retriever import MockRetriever
from ranker import MockRanker

class ContextReasoner:
    """Simulates the 'Reason' stage of CGR^3."""
    
    def __init__(self, retriever: MockRetriever, ranker: MockRanker, beam_width: int = 3, max_hops: int = 2):
        self.retriever = retriever
        self.ranker = ranker
        self.beam_width = beam_width
        self.max_hops = max_hops

    def reason(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs iterative graph traversal (multi-hop) to gather context.
        Uses a simplified beam search strategy.
        """
        # Step 1: Initialization (Entities extraction)
        topic_entities = self.retriever.retrieve_topic_entities(query)
        if not topic_entities:
            return []

        gathered_context = []
        current_beam = [(entity.entity_id, 0) for entity in topic_entities] # (entity_id, hop_count)
        visited = set()

        # Step 2: Multi-hop Traversal
        for hop in range(self.max_hops):
            next_beam_candidates = []
            
            for entity_id, _ in current_beam:
                if entity_id in visited:
                    continue
                visited.add(entity_id)
                
                # Retrieval: Fetch immediate triples
                triples = self.retriever.fetch_triples_and_context(entity_id)
                
                # Ranking: Score and filter triples
                ranked_triples = self.ranker.rank_candidates(query, triples)
                
                # Keep top-K based on beam width
                for triple, score in ranked_triples[:self.beam_width]:
                    gathered_context.append({
                        "hop": hop + 1,
                        "data": triple,
                        "score": score
                    })
                    next_beam_candidates.append((triple['tail'].entity_id, hop + 1))
            
            if not next_beam_candidates:
                break
                
            # Update beam for next hop
            current_beam = next_beam_candidates[:self.beam_width]

        return gathered_context
