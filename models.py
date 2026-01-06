from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class EntityContext:
    """Stores attributes and metadata for an entity."""
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    external_links: List[str] = field(default_factory=list)
    multimodal_context: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RelationContext:
    """Stores contextual information for a relationship/fact."""
    temporal: Optional[Dict[str, Any]] = None  # e.g., {"start": "2010", "end": "2012"}
    geographic: Optional[str] = None
    quantitative: Optional[Dict[str, Any]] = None
    provenance: List[str] = field(default_factory=list)  # Supporting sentences
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextNode:
    """Represents an entity E = (e, ec)."""
    entity_id: str
    label: str
    context: EntityContext = field(default_factory=EntityContext)

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if not isinstance(other, ContextNode):
            return False
        return self.entity_id == other.entity_id

@dataclass
class ContextEdge:
    """Represents a fact (h, r, t, rc)."""
    head: str  # Entity ID
    relation: str
    tail: str  # Entity ID
    context: RelationContext = field(default_factory=RelationContext)
