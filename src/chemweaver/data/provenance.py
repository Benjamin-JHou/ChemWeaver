"""
Virtual Screening Standard Schema (VSSS) - Provenance and Lineage Tracking
===========================================================================

FAIR-compliant provenance tracking with full experiment lineage graphs.

Implements W3C PROV-O standards with extensions for virtual screening.
Supports:
- Persistent compound identifiers (DOI-compatible)
- Complete lineage graph tracking
- W3C PROV-O serialization
- Experimental validation linkage

Author: VSSS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4


class ProvenanceEntityType(Enum):
    """Types of entities in provenance graph."""
    COMPOUND = "compound"
    TARGET = "target"
    SCREENING_RESULT = "screening_result"
    MODEL = "model"
    DATASET = "dataset"
    EXPERIMENT = "experiment"
    PUBLICATION = "publication"
    ASSAY = "assay"
    VALIDATION_RESULT = "validation_result"


class ProvenanceActivityType(Enum):
    """Types of activities in provenance graph."""
    PREPROCESSING = "preprocessing"
    DOCKING = "docking"
    SCORING = "scoring"
    AI_INFERENCE = "ai_inference"
    CONSENSUS = "consensus"
    FILTERING = "filtering"
    RANKING = "ranking"
    VALIDATION = "validation"
    PUBLICATION = "publication"


class ProvenanceAgentType(Enum):
    """Types of agents in provenance graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    SOFTWARE = "software"
    WORKFLOW = "workflow"
    MODEL = "model"


@dataclass
class PersistentIdentifier:
    """
    Persistent identifier with DOI compatibility.
    
    Supports multiple PID schemes:
    - DOI (Digital Object Identifier)
    - URI (Uniform Resource Identifier)
    - UUID (Universally Unique Identifier)
    - InChIKey (for compounds)
    """
    identifier_type: str  # doi, uri, uuid, inchikey, custom
    identifier_value: str
    
    # Resolution information
    resolver_url: Optional[str] = None
    
    # Registration information
    registered_at: Optional[datetime] = None
    registered_by: Optional[str] = None
    
    # Status
    is_resolvable: bool = True
    
    def to_uri(self) -> str:
        """Convert to resolvable URI."""
        if self.identifier_type == 'doi':
            return f"https://doi.org/{self.identifier_value}"
        elif self.identifier_type == 'uri':
            return self.identifier_value
        elif self.identifier_type == 'uuid':
            return f"urn:uuid:{self.identifier_value}"
        else:
            return f"vsss:{self.identifier_type}:{self.identifier_value}"


@dataclass
class ProvenanceAgent:
    """Agent in provenance graph (person, organization, software)."""
    agent_id: UUID
    agent_type: ProvenanceAgentType
    name: str
    
    # Contact information
    email: Optional[str] = None
    orcid: Optional[str] = None  # ORCID for researchers
    organization: Optional[str] = None
    
    # Software-specific
    software_version: Optional[str] = None
    software_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "@id": f"agent:{self.agent_id}",
            "@type": self.agent_type.value,
            "name": self.name,
            "email": self.email,
            "orcid": self.orcid,
            "organization": self.organization
        }


@dataclass
class ProvenanceEntity:
    """Entity in provenance graph (dataset, compound, result, etc.)."""
    entity_id: UUID
    entity_type: ProvenanceEntityType
    
    # Identifiers
    persistent_ids: List[PersistentIdentifier] = field(default_factory=list)
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    
    # External references
    external_references: Dict[str, str] = field(default_factory=dict)
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def get_primary_identifier(self) -> str:
        """Get primary persistent identifier as string."""
        if self.persistent_ids:
            return self.persistent_ids[0].identifier_value
        return str(self.entity_id)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "@id": f"entity:{self.entity_id}",
            "@type": self.entity_type.value,
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes
        }
        
        if self.persistent_ids:
            result["persistent_identifiers"] = [
                {
                    "type": pid.identifier_type,
                    "value": pid.identifier_value,
                    "uri": pid.to_uri()
                }
                for pid in self.persistent_ids
            ]
        
        return result


@dataclass
class ProvenanceActivity:
    """Activity in provenance graph (processing step, experiment)."""
    activity_id: UUID
    activity_type: ProvenanceActivityType
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Agent responsible
    agent_id: Optional[UUID] = None
    
    # Execution context reference
    execution_context_id: Optional[UUID] = None
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def duration_seconds(self) -> Optional[float]:
        """Calculate activity duration in seconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "@id": f"activity:{self.activity_id}",
            "@type": self.activity_type.value,
            "name": self.name,
            "description": self.description,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "parameters": self.parameters
        }


@dataclass
class ProvenanceRelation:
    """Relationship between provenance elements."""
    relation_type: str  # wasGeneratedBy, used, wasAttributedTo, wasDerivedFrom, etc.
    source_id: str
    target_id: str
    
    # Optional qualifier
    role: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "@type": self.relation_type,
            "source": self.source_id,
            "target": self.target_id,
            "role": self.role,
            "attributes": self.attributes
        }


@dataclass
class LineageNode:
    """Node in the screening lineage graph."""
    node_id: str
    node_type: str  # compound, preprocessing, screening, scoring, ai_inference, filtering, ranking
    
    # References
    entity_id: Optional[UUID] = None
    activity_id: Optional[UUID] = None
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    # Metrics
    input_count: Optional[int] = None
    output_count: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "activity_id": str(self.activity_id) if self.activity_id else None,
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "processing_time_seconds": self.processing_time_seconds,
            "attributes": self.attributes
        }


@dataclass
class LineageEdge:
    """Edge in the screening lineage graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    
    # Edge type
    edge_type: str  # transforms_to, filters_to, derives_from, scores, etc.
    
    # Filtering/selection information
    selection_criteria: Optional[str] = None
    filter_threshold: Optional[float] = None
    
    # Statistics
    items_processed: Optional[int] = None
    items_selected: Optional[int] = None
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.edge_id,
            "source": self.source_node_id,
            "target": self.target_node_id,
            "type": self.edge_type,
            "selection_criteria": self.selection_criteria,
            "filter_threshold": self.filter_threshold,
            "items_processed": self.items_processed,
            "items_selected": self.items_selected,
            "attributes": self.attributes
        }


class ScreeningLineageGraph:
    """
    Directed Acyclic Graph (DAG) tracking screening lineage.
    
    Captures complete traceability from input compounds through
    all screening stages to final ranked hits.
    """
    
    def __init__(self, graph_id: Optional[UUID] = None, name: Optional[str] = None):
        self.graph_id = graph_id or uuid4()
        self.name = name or f"lineage_graph_{self.graph_id}"
        
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, LineageEdge] = {}
        self.node_edges: Dict[str, Set[str]] = {}  # node_id -> set of edge_ids
        
        # Root and terminal nodes
        self.root_nodes: Set[str] = set()
        self.terminal_nodes: Set[str] = set()
    
    def add_node(self, node: LineageNode) -> None:
        """Add a node to the lineage graph."""
        self.nodes[node.node_id] = node
        self.node_edges[node.node_id] = set()
        
        # Initially assume it's a root node
        self.root_nodes.add(node.node_id)
        self.terminal_nodes.add(node.node_id)
    
    def add_edge(self, edge: LineageEdge) -> None:
        """Add an edge to the lineage graph."""
        self.edges[edge.edge_id] = edge
        
        # Update node connections
        self.node_edges[edge.source_node_id].add(edge.edge_id)
        
        # Update root/terminal status
        self.terminal_nodes.discard(edge.source_node_id)
        self.root_nodes.discard(edge.target_node_id)
    
    def get_upstream_nodes(self, node_id: str) -> List[LineageNode]:
        """Get all upstream nodes (predecessors) for a given node."""
        upstream = []
        visited = set()
        
        def traverse(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            
            for edge in self.edges.values():
                if edge.target_node_id == current_id:
                    upstream.append(self.nodes[edge.source_node_id])
                    traverse(edge.source_node_id)
        
        traverse(node_id)
        return upstream
    
    def get_downstream_nodes(self, node_id: str) -> List[LineageNode]:
        """Get all downstream nodes (successors) for a given node."""
        downstream = []
        visited = set()
        
        def traverse(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)
            
            for edge in self.edges.values():
                if edge.source_node_id == current_id:
                    downstream.append(self.nodes[edge.target_node_id])
                    traverse(edge.target_node_id)
        
        traverse(node_id)
        return downstream
    
    def get_lineage_for_compound(self, compound_id: str) -> List[LineageNode]:
        """Get complete lineage for a specific compound."""
        # Find node containing this compound
        compound_node = None
        for node in self.nodes.values():
            if node.attributes.get('compound_id') == compound_id:
                compound_node = node
                break
        
        if compound_node is None:
            return []
        
        # Get full ancestry
        return self.get_upstream_nodes(compound_node.node_id) + [compound_node]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize lineage graph to dictionary."""
        return {
            "graph_id": str(self.graph_id),
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "root_nodes": list(self.root_nodes),
            "terminal_nodes": list(self.terminal_nodes),
            "metadata": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "created_at": datetime.utcnow().isoformat()
            }
        }
    
    def to_networkx(self):
        """Convert to NetworkX graph for analysis."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required for graph analysis. Install with: pip install networkx")
        
        G = nx.DiGraph()
        
        for node in self.nodes.values():
            G.add_node(node.node_id, **node.to_dict())
        
        for edge in self.edges.values():
            G.add_edge(edge.source_node_id, edge.target_node_id, **edge.to_dict())
        
        return G
    
    def export_prov_json(self) -> Dict[str, Any]:
        """Export as W3C PROV-JSON format."""
        prov_doc = {
            "prefix": {
                "vsss": "http://vsss.org/ns/",
                "prov": "http://www.w3.org/ns/prov#"
            },
            "entity": {},
            "activity": {},
            "agent": {},
            "wasGeneratedBy": {},
            "used": {},
            "wasDerivedFrom": {},
            "wasAttributedTo": {}
        }
        
        # Convert nodes to PROV entities and activities
        for node in self.nodes.values():
            if node.entity_id:
                prov_doc["entity"][f"vsss:entity_{node.node_id}"] = {
                    "prov:label": node.name or node.node_id,
                    "prov:type": node.node_type
                }
            
            if node.activity_id:
                prov_doc["activity"][f"vsss:activity_{node.node_id}"] = {
                    "prov:label": node.name or node.node_id,
                    "prov:type": node.node_type,
                    "prov:startTime": node.timestamp.isoformat() if node.timestamp else None
                }
        
        # Convert edges to PROV relations
        for edge in self.edges.values():
            relation_id = f"vsss:relation_{edge.edge_id}"
            
            if edge.edge_type == "transforms_to":
                prov_doc["wasDerivedFrom"][relation_id] = {
                    "prov:generatedEntity": f"vsss:entity_{edge.target_node_id}",
                    "prov:usedEntity": f"vsss:entity_{edge.source_node_id}"
                }
        
        return prov_doc


class ProvenanceStore:
    """
    Store for provenance and lineage information.
    
    Manages:
    - Persistent identifiers
    - Complete provenance records
    - Lineage graphs
    - FAIR compliance metadata
    """
    
    def __init__(self):
        self.entities: Dict[UUID, ProvenanceEntity] = {}
        self.activities: Dict[UUID, ProvenanceActivity] = {}
        self.agents: Dict[UUID, ProvenanceAgent] = {}
        self.lineage_graphs: Dict[UUID, ScreeningLineageGraph] = {}
    
    def register_entity(
        self,
        entity_type: ProvenanceEntityType,
        persistent_ids: Optional[List[PersistentIdentifier]] = None,
        name: Optional[str] = None,
        **attributes
    ) -> ProvenanceEntity:
        """Register a new entity with provenance tracking."""
        entity = ProvenanceEntity(
            entity_id=uuid4(),
            entity_type=entity_type,
            persistent_ids=persistent_ids or [],
            name=name,
            attributes=attributes,
            created_at=datetime.utcnow()
        )
        self.entities[entity.entity_id] = entity
        return entity
    
    def register_activity(
        self,
        activity_type: ProvenanceActivityType,
        name: Optional[str] = None,
        agent_id: Optional[UUID] = None,
        **parameters
    ) -> ProvenanceActivity:
        """Register a new activity."""
        activity = ProvenanceActivity(
            activity_id=uuid4(),
            activity_type=activity_type,
            name=name,
            agent_id=agent_id,
            parameters=parameters,
            started_at=datetime.utcnow()
        )
        self.activities[activity.activity_id] = activity
        return activity
    
    def create_lineage_graph(self, name: Optional[str] = None) -> ScreeningLineageGraph:
        """Create a new lineage graph."""
        graph = ScreeningLineageGraph(name=name)
        self.lineage_graphs[graph.graph_id] = graph
        return graph
    
    def export_fair_metadata(self, entity_id: UUID) -> Dict[str, Any]:
        """
        Export FAIR-compliant metadata for an entity.
        
        Implements FAIR principles:
        - Findable: Persistent identifiers, rich metadata
        - Accessible: Resolvable URLs, access protocols
        - Interoperable: Standard formats, vocabularies
        - Reusable: Licenses, provenance, domain information
        """
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not found")
        
        return {
            "@context": {
                "vsss": "http://vsss.org/ns/",
                "schema": "http://schema.org/",
                "prov": "http://www.w3.org/ns/prov#",
                "dct": "http://purl.org/dc/terms/"
            },
            "@type": ["vsss:Entity", f"vsss:{entity.entity_type.value}"],
            "@id": entity.get_primary_identifier(),
            "dct:identifier": [
                {
                    "@type": "dct:Identifier",
                    "scheme": pid.identifier_type,
                    "value": pid.identifier_value
                }
                for pid in entity.persistent_ids
            ],
            "schema:name": entity.name,
            "schema:description": entity.description,
            "dct:created": entity.created_at.isoformat() if entity.created_at else None,
            "dct:type": entity.entity_type.value,
            "prov:wasAttributedTo": [],
            "vsss:attributes": entity.attributes
        }
    
    def generate_doi_metadata(self, entity_id: UUID) -> Dict[str, Any]:
        """Generate DataCite-compatible metadata for DOI registration."""
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not found")
        
        return {
            "identifiers": [
                {
                    "identifierType": pid.identifier_type.upper(),
                    "identifier": pid.identifier_value
                }
                for pid in entity.persistent_ids
            ],
            "titles": [
                {
                    "title": entity.name or "Untitled VSSS Entity"
                }
            ],
            "descriptions": [
                {
                    "description": entity.description or "",
                    "descriptionType": "Abstract"
                }
            ] if entity.description else [],
            "types": {
                "resourceTypeGeneral": "Dataset",
                "resourceType": f"VSSS {entity.entity_type.value}"
            },
            "subjects": [
                {
                    "subject": "virtual screening"
                },
                {
                    "subject": "computational chemistry"
                },
                {
                    "subject": "drug discovery"
                }
            ],
            "dates": [
                {
                    "date": entity.created_at.strftime("%Y-%m-%d") if entity.created_at else None,
                    "dateType": "Created"
                }
            ],
            "relatedIdentifiers": []
        }


class ExperimentalValidationLinkage:
    """
    Linkage layer for connecting computational predictions to
    experimental validation data.
    """
    
    @dataclass
    class AssayResult:
        """Individual assay measurement."""
        assay_id: str
        assay_type: str  # binding, functional, cellular, toxicity, etc.
        
        # Measurement
        value: float
        unit: str
        operator: str = "="  # =, <, >, <=, >=
        
        # Metadata
        compound_concentration: Optional[float] = None
        replicate_number: int = 1
        
        # Quality
        data_quality: str = "high"  # high, medium, low
        notes: Optional[str] = None
    
    @dataclass
    class ValidationEntry:
        """Complete validation record for a compound."""
        entry_id: UUID
        compound_id: UUID
        
        # Computational prediction
        screening_result_id: Optional[UUID] = None
        predicted_activity: Optional[float] = None
        prediction_confidence: Optional[float] = None
        
        # Experimental results
        assay_results: List[AssayResult] = field(default_factory=list)
        
        # Validation metadata
        lab: Optional[str] = None
        assay_date: Optional[datetime] = None
        technician: Optional[str] = None
        
        # Cross-validation
        cross_validation_available: bool = False
        cross_validation_results: List[AssayResult] = field(default_factory=list)
    
    def __init__(self):
        self.validation_entries: Dict[UUID, ValidationEntry] = {}
    
    def link_prediction_to_experiment(
        self,
        compound_id: UUID,
        screening_result_id: UUID,
        assay_results: List[AssayResult],
        lab: Optional[str] = None
    ) -> ValidationEntry:
        """Link a computational prediction to experimental validation."""
        entry = self.ValidationEntry(
            entry_id=uuid4(),
            compound_id=compound_id,
            screening_result_id=screening_result_id,
            assay_results=assay_results,
            lab=lab,
            assay_date=datetime.utcnow()
        )
        self.validation_entries[entry.entry_id] = entry
        return entry
    
    def calculate_validation_metrics(
        self,
        experiment_id: UUID
    ) -> Dict[str, float]:
        """
        Calculate validation metrics for an experiment.
        
        Returns metrics like:
        - Hit rate
        - Enrichment factor
        - Correlation between predicted and experimental
        """
        # Filter entries for this experiment
        entries = [
            e for e in self.validation_entries.values()
            # Would need to join with screening results to filter by experiment
        ]
        
        if not entries:
            return {}
        
        # Calculate metrics
        total_validated = len(entries)
        hits = sum(
            1 for e in entries
            if any(r.value < 10000 for r in e.assay_results if r.unit == "nM")
        )
        
        return {
            "total_validated": total_validated,
            "confirmed_hits": hits,
            "hit_rate": hits / total_validated if total_validated > 0 else 0.0,
            "validation_rate": total_validated / len(self.validation_entries) if self.validation_entries else 0.0
        }
