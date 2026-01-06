# Enhanced Context Graph Prototype (ToG-3 + Neo4j)

Neuro-symbolic Context Graph implementation using CGR3, ToG-3 (MACER), and Neo4j for deep, context-aware reasoning.

## Architectural Philosophy (AI Field Theory)

This implementation is grounded in the three pillars of AI Field Theory:
1.  **Continuity (The Flow)**: Replacing discrete tokens with a multi-layered information web (Chunks $\to$ Triplets $\to$ Communities).
2.  **Attractors (The Gravity)**: Establishing stability through deep contextual metadata (Temporal, Geographic, Provenance).
3.  **Resonance (The Vibration)**: Utilizing the **MACER** loop to filter noise and amplify high-fidelity evidence via iterative ranking and path pruning.

## Core Features

### 1. Heterogeneous Graph Index (MACER)
The graph structure is now multi-level, combining different granularities of information:
- **Chunks (C)**: Raw text segments providing granular evidence.
- **Triplets (T)**: Reified semantic facts $(s, p, o)$ with detailed context (temporal, provenance).
- **Communities (M)**: High-level summaries of entity clusters for global topical context.

### 2. MACER Reasoning Loop
The reasoning logic implements **Multi-Agent Context Evolution and Retrieval**:
- **Constructor Agent**: Dynamically builds task-specific sub-graphs from the heterogeneous index.
- **Reflector Agent**: Evaluates the gathered context against the query and triggers "evolution" (sub-querying) if information is insufficient.
- **Dual-Evolution**: Iteratively expands the available evidence until the reasoning path is complete.

### 3. Neo4j Backend
- **Advanced Persistence**: Every fact is stored as a first-class citizen with rich properties.
- **Scalable Retrieval**: Leverages Neo4j's graph querying capabilities for multi-hop discovery.

## Getting Started

### Installation
```bash
git clone https://github.com/bionicbutterfly13/context-graph-proto.git
cd context-graph-proto
pip install neo4j pytest
```

### Usage
1. **Ingest Sample Data**:
   Populate your Neo4j instance with context-rich data:
   ```bash
   python main.py --query "initialization" --ingest --uri bolt://localhost:7687 --user neo4j --password your_password
   ```

2. **Run ToG-3 Reasoner**:
   Execute the iterative reasoning loop:
   ```bash
   python main.py --query "Tell me about Albert Einstein in 1921." --password your_password
   ```

## Project Structure
- `neo4j_provider.py`: Neo4j connection and schema management.
- `models.py`: Heterogeneous data structures (Chunk, Triplet, Community).
- `reasoner.py`: MACER Agent Loop implementation.
- `retriever.py`: Multi-level context fetching.

## References
- **"Context Graph"** original paper.
- **"Think-on-Graph 3.0" (ToG-3)** paper.
