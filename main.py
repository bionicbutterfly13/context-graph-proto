from llm_util import LLMInterface
from reasoner import MACERReasoner

def ingest_sample_data(provider: Neo4jContextGraph):
    """Ingests heterogeneous context graph data (Quadruples)."""
    print("Ingesting sample CGR3 data into Neo4j...")
    provider.initialize_schema()

    # 1. Chunks (Source Text / Evidence)
    c1 = ChunkNode("C1", "Albert Einstein was awarded the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.")
    c2 = ChunkNode("C2", "The Nobel Committee for Physics awarded Einstein the prize in late 1922.")
    provider.add_chunk(c1)
    provider.add_chunk(c2)

    # 2. Entities (with EC)
    einstein = ContextNode("Q937", "Albert Einstein", context=EntityContext(
        metadata={"description": "Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time."}
    ))
    nobel = ContextNode("Q38104", "Nobel Prize in Physics", context=EntityContext(
        metadata={"description": "The Nobel Prize in Physics is a yearly award given by the Royal Swedish Academy of Sciences."}
    ))
    
    provider.add_entity(einstein)
    provider.add_entity(nobel)

    # 3. Triplets with RC (Quadruples)
    # (h, r, t, rc)
    rc = RelationContext(
        temporal={"year": 1921, "announced": 1922},
        provenance=["Nobel Foundation", "Wikipedia"],
        confidence=1.0,
        details={"category": "Physics"}
    )
    edge1 = ContextEdge("Q937", "WON", "Q38104", context=rc)
    provider.add_triplet_with_context(edge1, chunk_ids=["C1", "C2"])

def main():
    parser = argparse.ArgumentParser(description="Context Graph (CGR3 Phase 3) Implementation")
    parser.add_argument("--query", type=str, required=True, help="User query to process")
    parser.add_argument("--ingest", action="store_true", help="Ingest sample data into Neo4j")
    
    # Neo4j Config
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    
    args = parser.parse_args()
    
    provider = Neo4jContextGraph(args.uri, args.user, args.password)
    
    if args.ingest:
        try:
            ingest_sample_data(provider)
            print("Ingestion complete.\n")
        except Exception as e:
            print(f"Ingestion failed: {e}")
            return

    # Initialize Components
    llm = LLMInterface()
    reasoner = MACERReasoner(provider, llm)
    
    print(f"Executing CGR3 Reasoning Loop for: '{args.query}'\n")
    
    # Run Retrieve-Rank-Reason
    results = reasoner.reason(args.query)
    
    if results["answer"]:
        print("\n--- Final Answer (Synthesized) ---")
        print(results["answer"])
    else:
        print("\nSufficiency Gate: Insufficient information to provide a verified answer.")
    
    print("\n--- Gathered Context ---")
    for ctx in results["final_context"]:
        tail = ctx['tail']
        tr = ctx['tr']
        print(f"Fact: {tr['relation']} -> {tail['label']}")
        print(f"  Temporal: {[f'{k}:{v}' for k,v in tr.items() if k.startswith('temporal')]}")
            
    provider.close()

if __name__ == "__main__":
    main()
