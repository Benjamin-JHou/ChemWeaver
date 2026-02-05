"""
CAS-VS (Compute-Adaptive Screening Virtual Screening)
Core Multi-Stage Screening Execution Graph

This module implements the dynamic DAG representing screening stages
as defined in the CAS-VS architecture specification.

Stages:
- Stage 0: Library Standardization
- Stage 1: Ultra-Fast Ligand-Based Filtering
- Stage 2: Pharmacophore/Shape Screening
- Stage 3: Docking Screening
- Stage 4: AI Rescoring/Activity Prediction

Author: CAS-VS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4


class ScreeningStageType(Enum):
    """Enumeration of screening stage types."""
    STANDARDIZATION = 0        # Stage 0: Library Standardization
    LIGAND_FILTERING = 1       # Stage 1: Ultra-Fast Ligand-Based Filtering
    PHARMACOPHORE = 2          # Stage 2: Pharmacophore/Shape Screening
    DOCKING = 3                # Stage 3: Docking Screening
    AI_RESCORING = 4           # Stage 4: AI Rescoring/Activity Prediction
    CUSTOM = 99                # Custom user-defined stage


class StageStatus(Enum):
    """Execution status of a screening stage."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CACHED = auto()


@dataclass
class StageMetrics:
    """Performance metrics for a screening stage."""
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    execution_time_seconds: float = 0.0
    cpu_hours: float = 0.0
    memory_peak_gb: float = 0.0
    io_read_gb: float = 0.0
    io_write_gb: float = 0.0
    
    # AI-specific metrics
    inference_count: int = 0
    inference_time_seconds: float = 0.0
    gpu_hours: float = 0.0
    
    @property
    def enrichment_factor(self) -> Optional[float]:
        """Calculate enrichment factor for this stage."""
        if self.input_count == 0:
            return None
        return self.input_count / max(self.output_count, 1)
    
    @property
    def filter_rate(self) -> Optional[float]:
        """Calculate filter rate (fraction removed)."""
        if self.input_count == 0:
            return None
        return self.filtered_count / self.input_count
    
    @property
    def throughput_per_second(self) -> float:
        """Calculate throughput in compounds per second."""
        if self.execution_time_seconds == 0:
            return 0.0
        return self.input_count / self.execution_time_seconds


@dataclass
class StageConfiguration:
    """Configuration parameters for a screening stage."""
    
    # Stage identification
    stage_id: UUID = field(default_factory=uuid4)
    stage_type: ScreeningStageType = ScreeningStageType.CUSTOM
    stage_name: str = "unnamed_stage"
    stage_order: int = 0
    
    # Control parameters
    enabled: bool = True
    required: bool = False  # If True, stage cannot be skipped
    
    # Filtering thresholds
    filter_threshold: Optional[float] = None  # Score threshold for passing
    top_n_selection: Optional[int] = None     # Select top N compounds
    top_percent_selection: Optional[float] = None  # Select top N%
    
    # Adaptive parameters
    adaptive_threshold: bool = True  # Allow CAS engine to adjust thresholds
    min_output_count: int = 100      # Minimum compounds to pass forward
    max_output_count: Optional[int] = None  # Maximum compounds (None = unlimited)
    
    # Resource requirements
    min_cpu_cores: int = 1
    min_memory_gb: float = 4.0
    requires_gpu: bool = False
    min_gpu_memory_gb: Optional[float] = None
    estimated_time_per_compound_seconds: float = 0.1
    
    # Tool configuration
    tool_name: Optional[str] = None
    tool_version: Optional[str] = None
    tool_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Caching
    enable_caching: bool = True
    cache_key: Optional[str] = None
    
    def estimate_resource_requirements(
        self, 
        input_compound_count: int
    ) -> Dict[str, Any]:
        """Estimate resource requirements for given input size."""
        estimated_time = input_compound_count * self.estimated_time_per_compound_seconds
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_cpu_hours": estimated_time * self.min_cpu_cores / 3600,
            "estimated_memory_gb": self.min_memory_gb,
            "estimated_compounds": input_compound_count
        }


class ScreeningStage(ABC):
    """
    Abstract base class for screening stages.
    
    Each stage implements the execute() method with specific logic
    for filtering, scoring, or transforming compounds.
    """
    
    def __init__(self, config: StageConfiguration):
        self.config = config
        self.status = StageStatus.PENDING
        self.metrics = StageMetrics()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        
    @abstractmethod
    def execute(
        self, 
        input_compounds: List[UUID],
        context: ExecutionContext
    ) -> Tuple[List[UUID], Dict[str, Any]]:
        """
        Execute the screening stage.
        
        Args:
            input_compounds: List of compound UUIDs to process
            context: Execution context with resources and configuration
            
        Returns:
            Tuple of (passing_compounds, metadata)
        """
        pass
    
    def apply_selection_strategy(
        self,
        scored_compounds: List[Tuple[UUID, float]],
        strategy: str = "threshold"
    ) -> List[UUID]:
        """
        Apply compound selection strategy based on configuration.
        
        Args:
            scored_compounds: List of (compound_id, score) tuples
            strategy: Selection strategy (threshold, top_n, top_percent, adaptive)
            
        Returns:
            List of selected compound UUIDs
        """
        if strategy == "threshold" and self.config.filter_threshold is not None:
            return [cid for cid, score in scored_compounds 
                   if score >= self.config.filter_threshold]
        
        elif strategy == "top_n" and self.config.top_n_selection is not None:
            sorted_compounds = sorted(scored_compounds, key=lambda x: x[1], reverse=True)
            return [cid for cid, _ in sorted_compounds[:self.config.top_n_selection]]
        
        elif strategy == "top_percent" and self.config.top_percent_selection is not None:
            n_select = max(1, int(len(scored_compounds) * self.config.top_percent_selection / 100))
            sorted_compounds = sorted(scored_compounds, key=lambda x: x[1], reverse=True)
            return [cid for cid, _ in sorted_compounds[:n_select]]
        
        else:
            # Return all if no strategy specified
            return [cid for cid, _ in scored_compounds]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage to dictionary."""
        return {
            "stage_id": str(self.config.stage_id),
            "stage_type": self.config.stage_type.name,
            "stage_name": self.config.stage_name,
            "stage_order": self.config.stage_order,
            "status": self.status.name,
            "metrics": {
                "input_count": self.metrics.input_count,
                "output_count": self.metrics.output_count,
                "execution_time_seconds": self.metrics.execution_time_seconds,
                "enrichment_factor": self.metrics.enrichment_factor,
                "filter_rate": self.metrics.filter_rate
            },
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error_message
        }


@dataclass
class ExecutionContext:
    """Context for stage execution including resources and configuration."""
    
    # Execution identifiers
    execution_id: UUID = field(default_factory=uuid4)
    workflow_id: Optional[UUID] = None
    
    # Resources
    available_cpu_cores: int = 1
    available_memory_gb: float = 8.0
    available_gpus: int = 0
    gpu_memory_gb: float = 0.0
    
    # Compute backend
    backend_type: str = "local"  # local, slurm, pbs, kubernetes, aws, gcp
    backend_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Time constraints
    time_limit_seconds: Optional[float] = None
    deadline: Optional[datetime] = None
    
    # Storage
    storage_path: str = "./casvs_data"
    use_shared_storage: bool = False
    
    # CAS integration
    cas_enabled: bool = True
    cas_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Checkpointing
    checkpoint_interval_seconds: int = 300
    resume_from_checkpoint: Optional[str] = None


class StandardizationStage(ScreeningStage):
    """Stage 0: Library Standardization"""
    
    def __init__(self, config: Optional[StageConfiguration] = None):
        if config is None:
            config = StageConfiguration(
                stage_type=ScreeningStageType.STANDARDIZATION,
                stage_name="Library Standardization",
                stage_order=0,
                required=True,
                estimated_time_per_compound_seconds=0.01
            )
        super().__init__(config)
    
    def execute(
        self, 
        input_compounds: List[UUID],
        context: ExecutionContext
    ) -> Tuple[List[UUID], Dict[str, Any]]:
        """
        Execute standardization:
        - Canonicalize SMILES
        - Remove duplicates
        - Generate descriptors
        - Validate structures
        """
        self.status = StageStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.metrics.input_count = len(input_compounds)
        
        # Standardization logic would go here
        # For now, simulate with passthrough
        output_compounds = input_compounds.copy()
        
        # Simulate processing
        import time
        processing_time = len(input_compounds) * self.config.estimated_time_per_compound_seconds
        time.sleep(min(processing_time, 0.1))  # Simulate but don't wait too long
        
        self.metrics.output_count = len(output_compounds)
        self.metrics.filtered_count = self.metrics.input_count - self.metrics.output_count
        self.metrics.execution_time_seconds = processing_time
        
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.utcnow()
        
        metadata = {
            "standardized": True,
            "duplicates_removed": 0,
            "descriptors_generated": True
        }
        
        return output_compounds, metadata


class LigandFilteringStage(ScreeningStage):
    """Stage 1: Ultra-Fast Ligand-Based Filtering"""
    
    def __init__(self, config: Optional[StageConfiguration] = None):
        if config is None:
            config = StageConfiguration(
                stage_type=ScreeningStageType.LIGAND_FILTERING,
                stage_name="Ligand-Based Filtering",
                stage_order=1,
                filter_threshold=0.7,
                top_percent_selection=10.0,
                estimated_time_per_compound_seconds=0.001,
                tool_name="fingerprint_similarity",
                tool_parameters={"fingerprint_type": "morgan", "radius": 2}
            )
        super().__init__(config)
    
    def execute(
        self, 
        input_compounds: List[UUID],
        context: ExecutionContext
    ) -> Tuple[List[UUID], Dict[str, Any]]:
        """
        Execute ligand-based filtering:
        - Fingerprint similarity search
        - Substructure filtering
        - Rule-based medicinal chemistry filters
        """
        self.status = StageStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.metrics.input_count = len(input_compounds)
        
        # Simulate scoring (in real implementation, would compute fingerprints)
        scored = [(cid, 0.5 + (i % 10) / 20) for i, cid in enumerate(input_compounds)]
        
        # Apply selection
        output_compounds = self.apply_selection_strategy(scored, strategy="top_percent")
        
        self.metrics.output_count = len(output_compounds)
        self.metrics.filtered_count = self.metrics.input_count - self.metrics.output_count
        
        processing_time = len(input_compounds) * self.config.estimated_time_per_compound_seconds
        self.metrics.execution_time_seconds = processing_time
        
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.utcnow()
        
        metadata = {
            "method": "fingerprint_similarity",
            "reference_compounds": [],
            "similarity_threshold": self.config.filter_threshold
        }
        
        return output_compounds, metadata


class PharmacophoreStage(ScreeningStage):
    """Stage 2: Pharmacophore/Shape Screening"""
    
    def __init__(self, config: Optional[StageConfiguration] = None):
        if config is None:
            config = StageConfiguration(
                stage_type=ScreeningStageType.PHARMACOPHORE,
                stage_name="Pharmacophore/Shape Screening",
                stage_order=2,
                filter_threshold=0.6,
                top_percent_selection=50.0,
                estimated_time_per_compound_seconds=0.05,
                tool_name="pharmacophore_alignment",
                tool_parameters={"pharmacophore_model": "default"}
            )
        super().__init__(config)
    
    def execute(
        self, 
        input_compounds: List[UUID],
        context: ExecutionContext
    ) -> Tuple[List[UUID], Dict[str, Any]]:
        """
        Execute pharmacophore/shape screening:
        - Pharmacophore mapping
        - Shape similarity calculation
        """
        self.status = StageStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.metrics.input_count = len(input_compounds)
        
        # Simulate pharmacophore screening
        scored = [(cid, 0.4 + (i % 8) / 20) for i, cid in enumerate(input_compounds)]
        output_compounds = self.apply_selection_strategy(scored, strategy="top_percent")
        
        self.metrics.output_count = len(output_compounds)
        self.metrics.filtered_count = self.metrics.input_count - self.metrics.output_count
        
        processing_time = len(input_compounds) * self.config.estimated_time_per_compound_seconds
        self.metrics.execution_time_seconds = processing_time
        
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.utcnow()
        
        metadata = {
            "method": "pharmacophore_alignment",
            "features_matched": []
        }
        
        return output_compounds, metadata


class DockingStage(ScreeningStage):
    """Stage 3: Docking Screening"""
    
    def __init__(self, config: Optional[StageConfiguration] = None):
        if config is None:
            config = StageConfiguration(
                stage_type=ScreeningStageType.DOCKING,
                stage_name="Molecular Docking",
                stage_order=3,
                filter_threshold=-7.0,
                top_n_selection=1000,
                estimated_time_per_compound_seconds=5.0,
                requires_gpu=False,
                tool_name="vina",
                tool_version="1.2.0",
                tool_parameters={
                    "exhaustiveness": 32,
                    "num_modes": 9,
                    "energy_range": 3
                }
            )
        super().__init__(config)
    
    def execute(
        self, 
        input_compounds: List[UUID],
        context: ExecutionContext
    ) -> Tuple[List[UUID], Dict[str, Any]]:
        """
        Execute molecular docking:
        - Run docking simulation
        - Score poses
        - Select top compounds
        """
        self.status = StageStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.metrics.input_count = len(input_compounds)
        
        # Simulate docking (negative scores are better)
        import random
        random.seed(42)
        scored = [(cid, -5.0 - random.random() * 5.0) for cid in input_compounds]
        
        output_compounds = self.apply_selection_strategy(scored, strategy="top_n")
        
        self.metrics.output_count = len(output_compounds)
        self.metrics.filtered_count = self.metrics.input_count - self.metrics.output_count
        
        processing_time = len(input_compounds) * self.config.estimated_time_per_compound_seconds
        self.metrics.execution_time_seconds = processing_time
        
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.utcnow()
        
        metadata = {
            "docking_tool": self.config.tool_name,
            "exhaustiveness": self.config.tool_parameters.get("exhaustiveness"),
            "average_score": sum(s for _, s in scored) / len(scored) if scored else 0
        }
        
        return output_compounds, metadata


class AIRescoringStage(ScreeningStage):
    """Stage 4: AI Rescoring/Activity Prediction"""
    
    def __init__(self, config: Optional[StageConfiguration] = None):
        if config is None:
            config = StageConfiguration(
                stage_type=ScreeningStageType.AI_RESCORING,
                stage_name="AI Rescoring",
                stage_order=4,
                filter_threshold=0.8,
                top_n_selection=100,
                estimated_time_per_compound_seconds=0.1,
                requires_gpu=True,
                min_gpu_memory_gb=4.0,
                tool_name="gnn_rescoring",
                tool_parameters={
                    "model_name": "screening_gnn_v3",
                    "batch_size": 32
                }
            )
        super().__init__(config)
    
    def execute(
        self, 
        input_compounds: List[UUID],
        context: ExecutionContext
    ) -> Tuple[List[UUID], Dict[str, Any]]:
        """
        Execute AI rescoring:
        - Run GNN prediction
        - Calculate activity probability
        - Uncertainty-aware ranking
        """
        self.status = StageStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.metrics.input_count = len(input_compounds)
        
        # Simulate AI inference
        import random
        random.seed(42)
        scored = [(cid, random.random()) for cid in input_compounds]
        
        output_compounds = self.apply_selection_strategy(scored, strategy="top_n")
        
        self.metrics.output_count = len(output_compounds)
        self.metrics.filtered_count = self.metrics.input_count - self.metrics.output_count
        self.metrics.inference_count = len(input_compounds)
        
        processing_time = len(input_compounds) * self.config.estimated_time_per_compound_seconds
        self.metrics.execution_time_seconds = processing_time
        self.metrics.inference_time_seconds = processing_time
        
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.utcnow()
        
        metadata = {
            "model": self.config.tool_parameters.get("model_name"),
            "predictions": len(input_compounds),
            "average_probability": sum(s for _, s in scored) / len(scored) if scored else 0
        }
        
        return output_compounds, metadata


class MultiStageScreeningGraph:
    """
    Multi-Stage Screening Execution Graph (MANDATORY COMPONENT)
    
    Manages the dynamic DAG of screening stages, enabling:
    - Stage sequencing and dependencies
    - Adaptive stage skipping/enabling
    - Checkpoint recovery
    - Metrics aggregation
    """
    
    def __init__(self, graph_id: Optional[UUID] = None, name: Optional[str] = None):
        self.graph_id = graph_id or uuid4()
        self.name = name or f"screening_graph_{self.graph_id}"
        
        self.stages: Dict[UUID, ScreeningStage] = {}
        self.stage_order: List[UUID] = []
        self.edges: List[Tuple[UUID, UUID]] = []  # (from_stage, to_stage)
        
        # Execution state
        self.current_stage_idx: int = 0
        self.status: StageStatus = StageStatus.PENDING
        self.global_metrics: Dict[str, Any] = {}
        
        # Checkpointing
        self.checkpoints: List[Dict[str, Any]] = []
    
    def add_stage(self, stage: ScreeningStage) -> UUID:
        """Add a stage to the graph."""
        self.stages[stage.config.stage_id] = stage
        self.stage_order.append(stage.config.stage_id)
        
        # Sort by stage order
        self.stage_order.sort(
            key=lambda sid: self.stages[sid].config.stage_order
        )
        
        return stage.config.stage_id
    
    def add_edge(self, from_stage_id: UUID, to_stage_id: UUID) -> None:
        """Add dependency edge between stages."""
        self.edges.append((from_stage_id, to_stage_id))
    
    def get_stage_sequence(self) -> List[ScreeningStage]:
        """Get stages in execution order."""
        return [self.stages[sid] for sid in self.stage_order]
    
    def get_enabled_stages(self) -> List[ScreeningStage]:
        """Get only enabled stages."""
        return [s for s in self.get_stage_sequence() if s.config.enabled]
    
    def can_skip_stage(self, stage_id: UUID) -> bool:
        """Check if a stage can be skipped based on CAS decisions."""
        stage = self.stages.get(stage_id)
        if stage is None:
            return False
        return not stage.config.required and not stage.config.enabled
    
    def get_upstream_stages(self, stage_id: UUID) -> List[ScreeningStage]:
        """Get all stages that must complete before this stage."""
        upstream_ids = [src for src, tgt in self.edges if tgt == stage_id]
        return [self.stages[sid] for sid in upstream_ids if sid in self.stages]
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all stages."""
        total_input = 0
        total_output = 0
        total_time = 0.0
        total_cpu_hours = 0.0
        
        for stage in self.stages.values():
            total_input = max(total_input, stage.metrics.input_count)
            total_output = stage.metrics.output_count if stage.metrics.output_count > 0 else total_output
            total_time += stage.metrics.execution_time_seconds
            total_cpu_hours += stage.metrics.cpu_hours
        
        return {
            "total_compounds_input": total_input,
            "total_compounds_output": total_output,
            "overall_enrichment": total_input / max(total_output, 1),
            "total_execution_time_seconds": total_time,
            "total_cpu_hours": total_cpu_hours,
            "stage_count": len(self.stages),
            "completed_stages": sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED)
        }
    
    def create_checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint for resumption."""
        checkpoint = {
            "checkpoint_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "current_stage_idx": self.current_stage_idx,
            "stage_states": {
                str(sid): {
                    "status": stage.status.name,
                    "metrics": stage.to_dict()["metrics"]
                }
                for sid, stage in self.stages.items()
            }
        }
        self.checkpoints.append(checkpoint)
        return checkpoint
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "graph_id": str(self.graph_id),
            "name": self.name,
            "status": self.status.name,
            "stages": [self.stages[sid].to_dict() for sid in self.stage_order],
            "global_metrics": self.aggregate_metrics(),
            "checkpoint_count": len(self.checkpoints)
        }
    
    @classmethod
    def create_default_pipeline(cls, name: Optional[str] = None) -> MultiStageScreeningGraph:
        """Factory method to create default 5-stage pipeline."""
        graph = cls(name=name or "default_casvs_pipeline")
        
        # Create all 5 stages
        stage_0 = StandardizationStage()
        stage_1 = LigandFilteringStage()
        stage_2 = PharmacophoreStage()
        stage_3 = DockingStage()
        stage_4 = AIRescoringStage()
        
        # Add to graph
        graph.add_stage(stage_0)
        graph.add_stage(stage_1)
        graph.add_stage(stage_2)
        graph.add_stage(stage_3)
        graph.add_stage(stage_4)
        
        # Add linear edges
        s_ids = graph.stage_order
        for i in range(len(s_ids) - 1):
            graph.add_edge(s_ids[i], s_ids[i + 1])
        
        return graph


# Type exports
__all__ = [
    'ScreeningStageType',
    'StageStatus', 
    'StageMetrics',
    'StageConfiguration',
    'ScreeningStage',
    'ExecutionContext',
    'StandardizationStage',
    'LigandFilteringStage',
    'PharmacophoreStage',
    'DockingStage',
    'AIRescoringStage',
    'MultiStageScreeningGraph'
]
