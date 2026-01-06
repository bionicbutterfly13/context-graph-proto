# Context Graph Prototype ($CGR^3$)

A prototypical implementation of the Context Graph architecture, leveraging the **CGR³ (Retrieval-Rank-Reason)** paradigm to enable multi-hop reasoning with rich temporal, geographic, and provenance-based contexts. Based on the original "Context Graph" paper.

## Core Paradigm: $CGR^3$

- **Retrieval**: Extracting topic entities and fetching immediate contextual triples (Entity Context $ec$ and Relation Context $rc$).
- **Rank**: Scoring candidate facts using LLM-inspired relevance metrics and context availability.
- **Reason**: Multi-hop graph traversal (beam search) to recursively gather evidence and answer complex queries across the graph.

## Key Features

- **Rich Context Models**: Support for temporal data (validity period), geographic location, provenance (supporting sentences), and quantitative attributes.
- **Multi-Hop Traversal**: Automated reasoning pipeline that explores the graph recursively using a configurable beam width.
- **Entity & Relation Contexts**: Formal implementation of $CG = \{ \mathcal{E}, \mathcal{R}, \mathcal{Q}, \mathcal{EC}, \mathcal{RC} \}$.
- **Python-Based Core**: Lightweight, extensible architecture using a custom in-memory graph store.

## Getting Started

### Prerequisites
- Python 3.8+
- `pytest` (for running tests)

### Installation
```bash
git clone https://github.com/bionicbutterfly13/context-graph-proto.git
cd context-graph-proto
```

### Usage
Run the prototype with a sample query:
```bash
python main.py --query "Tell me about Albert Einstein winning the Nobel Prize in 1921."
```

### Running Tests
```bash
PYTHONPATH=. pytest tests/test_context_graph.py
```

## Project Structure
- `models.py`: Data structures for Entity and Relation contexts.
- `graph.py`: In-memory Context Graph implementation.
- `retriever.py`: CGR³ Retrieval logic.
- `ranker.py`: CGR³ Ranking logic.
- `reasoner.py`: CGR³ Reasoning and multi-hop traversal.
- `main.py`: CLI entry point and sample dataset.

## Acknowledgments
Implemented based on the architectural principles outlined in the **"Context Graph"** paper.
