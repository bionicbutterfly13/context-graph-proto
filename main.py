import argparse
from models import ContextNode, ContextEdge, EntityContext, RelationContext
from graph import ContextGraph
from retriever import MockRetriever
from ranker import MockRanker
from reasoner import ContextReasoner

def build_sample_graph():
    cg = ContextGraph()
    
    # Entities
    einstein = ContextNode("Q937", "Albert Einstein", EntityContext(
        attributes={"occupation": "Physicist", "birth_place": "Ulm, Germany"},
        metadata={"description": "Theoretical physicist who developed the theory of relativity."}
    ))
    nobel_physics = ContextNode("Q38104", "Nobel Prize in Physics", EntityContext(
        attributes={"established": "1901"},
        metadata={"description": "Awarded annually by the Royal Swedish Academy of Sciences."}
    ))
    photoelectric_effect = ContextNode("Q12345", "Photoelectric Effect", EntityContext(
        attributes={"discovered": "1887"},
        metadata={"description": "Observation that many metals emit electrons when light shines upon them."}
    ))
    
    cg.add_node(einstein)
    cg.add_node(nobel_physics)
    cg.add_node(photoelectric_effect)
    
    # Relations with Context
    win_nobel = ContextEdge("Q937", "won", "Q38104", RelationContext(
        temporal={"year": "1921"},
        provenance=["He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics'."],
        confidence=1.0
    ))
    discovery = ContextEdge("Q937", "explained", "Q12345", RelationContext(
        temporal={"year": "1905"},
        provenance=["Einstein's 1905 paper on the photoelectric effect was pivotal in quantum theory."],
        confidence=1.0
    ))
    
    cg.add_edge(win_nobel)
    cg.add_edge(discovery)
    
    return cg

def main():
    parser = argparse.ArgumentParser(description="Context Graph CGR^3 Prototype")
    parser.add_argument("--query", type=str, required=True, help="User query to process")
    args = parser.parse_args()
    
    print(f"\nProcessing Query: '{args.query}'\n")
    
    # Initialize components
    graph = build_sample_graph()
    retriever = MockRetriever(graph)
    ranker = MockRanker()
    reasoner = ContextReasoner(retriever, ranker)
    
    # Run CGR^3 Flow
    results = reasoner.reason(args.query)
    
    if not results:
        print("No relevant context found in the graph.")
        return
        
    print("--- Found Contexts ---")
    for res in results:
        data = res['data']
        hop = res['hop']
        score = res['score']
        
        ctx = data['context']
        print(f"[Hop {hop}] {data['head'].label} --({data['relation']})--> {data['tail'].label} (Score: {score:.2f})")
        if ctx.temporal:
            print(f"  Temporal: {ctx.temporal}")
        if ctx.provenance:
            for line in ctx.provenance:
                print(f"  Provenance: {line}")
        print("-" * 20)

if __name__ == "__main__":
    main()
