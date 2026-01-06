import argparse
import os
from models import ContextNode, ContextEdge, EntityContext, RelationContext, ChunkNode, CommunityNode
from neo4j_provider import Neo4jContextGraph
from retriever import Neo4jRetriever
from reasoner import MACERReasoner

def ingest_sample_data(provider: Neo4jContextGraph):
    """Ingests heterogeneous context graph data based on TOG-3."""
    print("Ingesting sample TO-G3 data into Neo4j...")
    provider.initialize_schema()

    # 1. Chunks (Source Text)
    c1 = ChunkNode("C1", "The Nobel Prize in Physics 1921 was awarded to Albert Einstein for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.")
    provider.add_chunk(c1)

    # 2. Entities
    einstein = ContextNode("Q937", "Albert Einstein", context=EntityContext(
        attributes={"born": 1879, "died": 1955},
        metadata={"description": "Genius theoretical physicist."}
    ))
    nobel = ContextNode("Q38104", "Nobel Prize in Physics")
    photo_effect = ContextNode("Q12345", "Photoelectric Effect")
    
    provider.add_entity(einstein)
    provider.add_entity(nobel)
    provider.add_entity(photo_effect)

    # 3. Communities
    m1 = CommunityNode("M1", "Theoretical Physics", "A community of entities related to the development of quantum mechanics and relativity.", ["Q937", "Q38104", "Q12345"])
    provider.add_community(m1)

    # 4. Triplets (Reified with Chunk linkage)
    edge1 = ContextEdge("Q937", "WON", "Q38104", context=RelationContext(temporal={"year": 1921}))
    provider.add_triplet_with_context(edge1, chunk_ids=["C1"])

    edge2 = ContextEdge("Q937", "DISCOVERED", "Q12345", context=RelationContext(temporal={"year": 1905}))
    provider.add_triplet_with_context(edge2, chunk_ids=["C1"])

def main():
    parser = argparse.ArgumentParser(description="Enhanced Context Graph (ToG-3) Prototype")
    parser.add_argument("--query", type=str, required=True, help="User query to process")
    parser.add_argument("--ingest", action="store_true", help="Ingest sample data into Neo4j")
    
    # Neo4j Config
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"))
    
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
    reasoner = MACERReasoner(provider)
    
    print(f"Executing TOG-3 Reasoning for query: '{args.query}'\n")
    
    # Run MACER Loop
    # Note: For the prototype, we assume entity extraction is handled by finding labels in query
    results = reasoner.reason(args.query)
    
    print("\n--- Final Context Sub-graph (ToG-3) ---")
    for ctx in results["final_context"]:
        e = ctx['e']
        tr = ctx['tr']
        tail = ctx['tail']
        chunks = ctx['chunks']
        communities = ctx['communities']
        
        print(f"Fact: {e['label']} --[{tr['relation']}]--> {tail['label']}")
        if tr.get('temporal'):
             print(f"  Temporal: {tr['temporal']}")
        
        if chunks:
            print(f"  Supporting Chunks: {[c['id'] for c in chunks]}")
            for c in chunks:
                print(f"    - {c['content'][:100]}...")
                
        if communities:
            print(f"  Communities: {[m['label'] for m in communities]}")
            
    provider.close()

if __name__ == "__main__":
    main()
