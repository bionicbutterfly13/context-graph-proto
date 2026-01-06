from typing import List, Dict, Any

class LLMInterface:
    """
    A utility to handle LLM completions for Ranking and Reasoning.
    In this environment, it will use the system's reasoning capabilities.
    """
    
    def generate(self, prompt: str) -> str:
        """
        Sends the prompt to the LLM. 
        Note: In a local deployment, this would be an API call to Gemini or a local vLLM instance.
        """
        print(f"\n--- [LLM PROMPT START] ---\n{prompt}\n--- [LLM PROMPT END] ---\n")
        
        # Placeholder for real LLM integration (e.g., vertexai or openai)
        return "Y" # Defaulting to 'Y' for prototype flow

    def build_ranking_prompt(self, query: str, head_context: str, candidates: List[Dict[str, Any]]) -> str:
        """Appendix A.3: Context-aware Re-Ranking template."""
        prompt = f"Context-aware Re-Ranking:\n"
        prompt += f"The question is to predict the missing entity for the query: {query}\n"
        prompt += f"Head Entity Context: {head_context}\n"
        prompt += f"The list of candidate answers is:\n"
        for i, c in enumerate(candidates):
            prompt += f"[{i+1}] {c['name']}: {c['description']}\n"
        
        prompt += "\nInstruction: Sort the list to let the candidate answers which are more possible to be the true answer to the question be prior. "
        prompt += "Output the sorted order using the format '[most possible, second possible...]' and please start your response with 'The final order:'."
        return prompt

    def build_reasoning_prompt(self, query: str, head_context: str, top_candidate: Dict[str, Any]) -> str:
        """Appendix A.2: Context-aware Reasoning / Sufficiency Check template."""
        prompt = f"Context-aware Reasoning:\n"
        prompt += f"Here are some materials for you to refer to.\n"
        prompt += f"Primary Entity Context: {head_context}\n"
        prompt += f"Top Candidate Fact: {top_candidate['name']} ({top_candidate['description']})\n"
        prompt += f"Supporting Evidence: {top_candidate.get('evidence', 'No specific evidence provided.')}\n\n"
        
        prompt += f"The question is to predict the missing entity for: {query}.\n"
        prompt += "Instruction: Output all possible answers you can find IN THE MATERIALS using the format '[answer1, answerN]'. "
        prompt += "Please start your response with 'The possible answers:'. Do not output anything except the possible answers. "
        prompt += "If you cannot find any answer, please output some possible answers based on your own knowledge but explicitly mark them."
        return prompt

    def build_type_reasoning_prompt(self, test_triple: str, fewshot_triples: str) -> str:
        """Official CATS TYPE_REASON_PROMPT template."""
        return f"""Please determine whether the entities in the input triples are consistent in entity type with a set of known triples in the knowledge graph provided.
A set of known triples are:
{fewshot_triples}
The triple to be determined is:
{test_triple}
Please return 'Y' if the input triple is consistent in entity type, otherwise return 'N'. Do not say anything else except your determination."""

    def build_subgraph_reasoning_prompt(self, test_triple: str, neighbor_triples: str, reasoning_paths: str) -> str:
        """Official CATS SUBGRAPH_REASON_PROMPT template."""
        return f"""Please determine whether the relation in the input can be reliably inferred between the head and tail entities, based on a set of neighbor triples and reasoning paths from the knowledge graph.
A set of neighbor triples from the knowledge graph are:
{neighbor_triples}
A set of reasoning paths from the knowledge graph are:
{reasoning_paths}
The relation to be inferred is:
{test_triple}
Please return 'Y' if there is sufficient evidence from the knowledge graph to infer the relation, otherwise return 'N'. Do not say anything else except your determination."
"""

    def build_evolution_query_prompt(self, original_query: str, context_summary: str) -> str:
        """ToG-3 Dual-Evolution placeholder template."""
        return f"""Based on the original query: '{original_query}' and the current context gathered:
{context_summary}

Determine what information is missing. Generate a short sub-query to expand the context graph search radius."""
