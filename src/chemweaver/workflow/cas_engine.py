"""
CAS-VS Compute-Adaptive Screening Strategy Engine

The CAS Engine dynamically optimizes screening strategy based on available
compute resources, target characteristics, and library properties.

Key Features:
- Resource-aware stage selection
- Adaptive threshold optimization
- Cost-performance frontier modeling
- Real-time strategy adjustment

Author: CAS-VS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np


class TargetDifficulty(Enum):
    """Target difficulty classification for strategy optimization."""
    EASY = "easy"           # High enrichment, clear binding site
    MODERATE = "moderate"   # Standard difficulty
    HARD = "hard"           # Challenging target, difficult pocket
    VERY_HARD = "very_hard" # Intrinsically disordered, multiple sites


class ResourceClass(Enum):
    """Classification of available compute resources."""
    WORKSTATION = "workstation"      # Single machine, 1-8 cores
    WORKSTATION_GPU = "workstation_gpu"  # With GPU acceleration
    SMALL_CLUSTER = "small_cluster"  # 10-100 cores
    MEDIUM_CLUSTER = "medium_cluster"  # 100-1000 cores
    LARGE_CLUSTER = "large_cluster"   # 1000+ cores
    CLOUD = "cloud"                  # Auto-scaling cloud


@dataclass
class ResourceDescriptor:
    """
    Descriptor of available compute resources.
    
    Captures complete resource state for adaptive decision making.
    """
    
    # Resource identification
    resource_id: UUID = field(default_factory=uuid4)
    resource_class: ResourceClass = ResourceClass.WORKSTATION
    
    # CPU resources
    total_cpu_cores: int = 4
    available_cpu_cores: int = 4
    cpu_frequency_ghz: float = 2.5
    cpu_architecture: str = "x86_64"
    
    # Memory
    total_memory_gb: float = 16.0
    available_memory_gb: float = 16.0
    
    # GPU resources
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_model: Optional[str] = None
    gpu_memory_gb: float = 0.0
    cuda_version: Optional[str] = None
    
    # Storage
    storage_type: str = "local_ssd"  # local_ssd, network, object_storage
    storage_available_gb: float = 100.0
    storage_bandwidth_mbps: float = 500.0
    
    # Network
    network_bandwidth_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Time constraints
    wall_clock_limit_seconds: Optional[float] = None
    deadline: Optional[datetime] = None
    
    # Cost (for cloud environments)
    cost_per_cpu_hour: float = 0.0
    cost_per_gpu_hour: float = 0.0
    budget_limit_usd: Optional[float] = None
    
    @property
    def estimated_throughput_compounds_per_hour(self) -> float:
        """Estimate screening throughput based on resources."""
        base_throughput = self.available_cpu_cores * 1000  # 1000 compounds/core/hour
        
        if self.gpu_available:
            # GPU can significantly accelerate ML inference
            base_throughput += self.gpu_count * 10000
        
        return base_throughput
    
    def can_execute_stage(self, min_cpu: int, min_memory_gb: float, requires_gpu: bool = False) -> bool:
        """Check if resource can execute a stage with given requirements."""
        if self.available_cpu_cores < min_cpu:
            return False
        if self.available_memory_gb < min_memory_gb:
            return False
        if requires_gpu and not self.gpu_available:
            return False
        return True
    
    def estimate_stage_cost(self, cpu_hours: float, gpu_hours: float = 0.0) -> float:
        """Estimate monetary cost for executing a stage."""
        cost = cpu_hours * self.cost_per_cpu_hour
        cost += gpu_hours * self.cost_per_gpu_hour
        return cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": str(self.resource_id),
            "resource_class": self.resource_class.value,
            "cpu_cores": self.available_cpu_cores,
            "memory_gb": self.available_memory_gb,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "estimated_throughput": self.estimated_throughput_compounds_per_hour,
            "wall_clock_limit": self.wall_clock_limit_seconds
        }


@dataclass
class TargetProfile:
    """Profile of the screening target for strategy optimization."""
    
    target_id: UUID
    target_name: str
    
    # Target characteristics
    difficulty: TargetDifficulty = TargetDifficulty.MODERATE
    binding_site_defined: bool = True
    binding_site_size_aa: Optional[float] = None
    
    # Historical performance (from benchmarks)
    historical_hit_rate: Optional[float] = None
    historical_enrichment_factor: Optional[float] = None
    
    # Pocket characteristics
    pocket_druggability_score: Optional[float] = None  # 0-1
    pocket_flexibility: Optional[str] = None  # rigid, flexible, very_flexible
    
    # Known actives
    known_actives_count: int = 0
    known_scaffolds: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": str(self.target_id),
            "target_name": self.target_name,
            "difficulty": self.difficulty.value,
            "binding_site_defined": self.binding_site_defined,
            "historical_hit_rate": self.historical_hit_rate,
            "historical_ef": self.historical_enrichment_factor
        }


@dataclass
class LibraryProfile:
    """Profile of the compound library being screened."""
    
    library_id: str
    library_name: str
    total_compounds: int
    
    # Library characteristics
    average_molecular_weight: Optional[float] = None
    diversity_index: Optional[float] = None  # Shannon entropy or similar
    novelty_score: Optional[float] = None    # 0-1, vs. known compounds
    
    # Pre-computed properties
    has_descriptors: bool = False
    has_fingerprints: bool = False
    has_3d_conformers: bool = False
    
    # Known enrichment (from prior screens)
    prior_enrichment_factor: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "library_id": self.library_id,
            "library_name": self.library_name,
            "total_compounds": self.total_compounds,
            "diversity_index": self.diversity_index
        }


@dataclass
class AdaptiveDecision:
    """
    A decision made by the CAS engine.
    
    Captures all inputs and outputs for reproducibility.
    """
    
    decision_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Decision context
    stage_id: Optional[UUID] = None
    decision_type: str = ""  # threshold_adjustment, stage_skip, batch_size, etc.
    
    # Inputs
    resource_state: Optional[ResourceDescriptor] = None
    target_profile: Optional[TargetProfile] = None
    library_profile: Optional[LibraryProfile] = None
    
    # Decision outputs
    previous_value: Any = None
    new_value: Any = None
    rationale: str = ""
    
    # Confidence
    confidence_score: float = 0.5  # 0-1
    alternative_decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance prediction
    predicted_enrichment: Optional[float] = None
    predicted_cost_usd: Optional[float] = None
    predicted_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": str(self.decision_id),
            "timestamp": self.timestamp.isoformat(),
            "stage_id": str(self.stage_id) if self.stage_id else None,
            "decision_type": self.decision_type,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "rationale": self.rationale,
            "confidence": self.confidence_score
        }


@dataclass
class StageOptimizationResult:
    """Result of optimizing a single stage."""
    
    stage_id: UUID
    optimized: bool
    
    # Optimized parameters
    filter_threshold: Optional[float] = None
    top_n_selection: Optional[int] = None
    top_percent_selection: Optional[float] = None
    batch_size: int = 1000
    
    # Resource allocation
    allocated_cpu_cores: int = 1
    allocated_memory_gb: float = 4.0
    use_gpu: bool = False
    
    # Predictions
    predicted_output_count: int = 0
    predicted_execution_time_seconds: float = 0.0
    predicted_cost_usd: float = 0.0
    
    # Decisions made
    decisions: List[AdaptiveDecision] = field(default_factory=list)


class ComputeAdaptiveScreeningEngine:
    """
    Compute-Adaptive Screening Strategy (CAS) Engine
    
    The core algorithmic engine that dynamically optimizes screening
    strategy based on available compute resources, target characteristics,
    and library properties.
    
    Algorithm:
    1. Assess current resource state
    2. Estimate stage resource requirements
    3. Optimize stage parameters for cost-performance frontier
    4. Make adaptive decisions with full logging
    5. Continuously monitor and adjust
    """
    
    def __init__(self, 
                 strategy: str = "balanced",
                 optimization_target: str = "enrichment"):
        """
        Initialize CAS Engine.
        
        Args:
            strategy: Overall strategy (fast, balanced, thorough, cost_optimized)
            optimization_target: Primary target (enrichment, speed, cost, discovery)
        """
        self.engine_id = uuid4()
        self.strategy = strategy
        self.optimization_target = optimization_target
        
        # Decision history
        self.decisions: List[AdaptiveDecision] = []
        
        # Configuration
        self.config = {
            "min_stage_time_seconds": 60,
            "max_stage_time_seconds": 86400,  # 24 hours
            "target_enrichment_factor": 10.0,
            "max_cost_per_stage_usd": 1000.0,
            "enable_ml_optimization": True
        }
        
        # Benchmark data for decision making
        self.benchmark_data: Dict[str, Any] = {}
        
    def assess_resource_class(self, resources: ResourceDescriptor) -> ResourceClass:
        """Classify resources into predefined classes."""
        if resources.gpu_available and resources.total_cpu_cores <= 16:
            return ResourceClass.WORKSTATION_GPU
        elif resources.total_cpu_cores <= 8:
            return ResourceClass.WORKSTATION
        elif resources.total_cpu_cores <= 100:
            return ResourceClass.SMALL_CLUSTER
        elif resources.total_cpu_cores <= 1000:
            return ResourceClass.MEDIUM_CLUSTER
        elif resources.total_cpu_cores > 1000:
            return ResourceClass.LARGE_CLUSTER
        else:
            return ResourceClass.CLOUD
    
    def estimate_stage_requirements(
        self,
        stage_type: str,
        input_compounds: int,
        stage_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate resource requirements for a screening stage.
        
        Returns dictionary with estimated:
        - cpu_hours
        - memory_gb
        - gpu_hours (if applicable)
        - time_seconds
        - cost_usd
        """
        time_per_compound = stage_config.get("time_per_compound_seconds", 1.0)
        total_time = input_compounds * time_per_compound
        
        cpu_cores = stage_config.get("min_cpu_cores", 1)
        cpu_hours = (total_time / 3600) * cpu_cores
        
        memory_gb = stage_config.get("min_memory_gb", 4.0)
        
        gpu_hours = 0.0
        if stage_config.get("requires_gpu", False):
            gpu_hours = total_time / 3600
        
        return {
            "cpu_hours": cpu_hours,
            "memory_gb": memory_gb,
            "gpu_hours": gpu_hours,
            "time_seconds": total_time,
            "parallelizable": stage_config.get("parallelizable", True)
        }
    
    def optimize_stage_parameters(
        self,
        stage_id: UUID,
        stage_type: str,
        input_compounds: int,
        resources: ResourceDescriptor,
        target: TargetProfile,
        library: LibraryProfile,
        current_config: Dict[str, Any]
    ) -> StageOptimizationResult:
        """
        Optimize parameters for a single screening stage.
        
        Considers:
        - Available resources
        - Target difficulty
        - Library size and properties
        - Strategy (fast/balanced/thorough)
        - Optimization target (enrichment/speed/cost)
        """
        result = StageOptimizationResult(
            stage_id=stage_id,
            optimized=True
        )
        
        decisions = []
        
        # Strategy-based threshold adjustment
        if self.strategy == "fast":
            # Aggressive filtering to reduce compounds quickly
            new_threshold = 0.8
            new_top_percent = 5.0
            rationale = "Fast strategy: aggressive filtering for speed"
        elif self.strategy == "thorough":
            # Conservative filtering to retain more compounds
            new_threshold = 0.5
            new_top_percent = 20.0
            rationale = "Thorough strategy: conservative filtering for discovery"
        else:  # balanced
            new_threshold = 0.7
            new_top_percent = 10.0
            rationale = "Balanced strategy: moderate filtering"
        
        # Adjust based on target difficulty
        if target.difficulty == TargetDifficulty.HARD:
            # For hard targets, be more conservative
            new_top_percent *= 1.5
            rationale += ", adjusted for hard target"
        elif target.difficulty == TargetDifficulty.EASY:
            # For easy targets, can be more aggressive
            new_top_percent *= 0.8
            rationale += ", adjusted for easy target"
        
        # Adjust based on library size
        if library.total_compounds > 1000000:
            # Large libraries need more aggressive filtering
            new_top_percent *= 0.7
            rationale += ", adjusted for large library"
        
        # Create threshold adjustment decision
        threshold_decision = AdaptiveDecision(
            stage_id=stage_id,
            decision_type="threshold_adjustment",
            previous_value=current_config.get("filter_threshold"),
            new_value=new_threshold,
            resource_state=resources,
            target_profile=target,
            library_profile=library,
            rationale=rationale,
            confidence_score=0.75
        )
        decisions.append(threshold_decision)
        
        result.filter_threshold = new_threshold
        result.top_percent_selection = new_top_percent
        
        # Allocate resources
        if resources.gpu_available and current_config.get("requires_gpu", False):
            result.use_gpu = True
            result.allocated_cpu_cores = min(4, resources.available_cpu_cores)
        else:
            result.use_gpu = False
            result.allocated_cpu_cores = min(8, resources.available_cpu_cores)
        
        result.allocated_memory_gb = current_config.get("min_memory_gb", 4.0)
        
        # Estimate output count
        result.predicted_output_count = int(input_compounds * new_top_percent / 100)
        result.predicted_output_count = max(100, result.predicted_output_count)  # Min 100
        
        # Estimate time
        time_per_compound = current_config.get("time_per_compound_seconds", 1.0)
        result.predicted_execution_time_seconds = input_compounds * time_per_compound / result.allocated_cpu_cores
        
        # Estimate cost
        cpu_hours = result.predicted_execution_time_seconds / 3600 * result.allocated_cpu_cores
        gpu_hours = cpu_hours if result.use_gpu else 0.0
        result.predicted_cost_usd = resources.estimate_stage_cost(cpu_hours, gpu_hours)
        
        result.decisions = decisions
        self.decisions.extend(decisions)
        
        return result
    
    def decide_stage_skip(
        self,
        stage_id: UUID,
        stage_type: str,
        resources: ResourceDescriptor,
        target: TargetProfile,
        time_remaining_seconds: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Decide whether to skip a stage based on resource constraints.
        
        Returns (should_skip, rationale)
        """
        # Never skip required stages
        if stage_type == "standardization":
            return False, "Required stage cannot be skipped"
        
        # Skip if insufficient time
        if time_remaining_seconds is not None and time_remaining_seconds < 3600:
            if stage_type in ["pharmacophore", "ai_rescoring"]:
                return True, f"Insufficient time ({time_remaining_seconds}s) for {stage_type}"
        
        # Skip docking for easy targets if resources limited
        if stage_type == "docking" and target.difficulty == TargetDifficulty.EASY:
            if resources.resource_class == ResourceClass.WORKSTATION:
                return True, "Skipping docking for easy target on limited resources"
        
        return False, "Stage should be executed"
    
    def optimize_batch_size(
        self,
        total_compounds: int,
        resources: ResourceDescriptor,
        memory_per_compound_mb: float = 10.0
    ) -> int:
        """
        Optimize batch size for parallel processing.
        
        Considers:
        - Available memory
        - CPU cores
        - I/O bandwidth
        """
        # Memory-based limit
        available_memory_mb = resources.available_memory_gb * 1024
        memory_based_batch = int(available_memory_mb / memory_per_compound_mb / 2)  # 50% buffer
        
        # CPU-based limit (aim for 10 batches per core)
        cpu_based_batch = max(100, int(total_compounds / (resources.available_cpu_cores * 10)))
        
        # Take minimum
        optimal_batch = min(memory_based_batch, cpu_based_batch, 10000)  # Max 10K
        optimal_batch = max(optimal_batch, 100)  # Min 100
        
        return optimal_batch
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get complete decision history for reproducibility."""
        return [d.to_dict() for d in self.decisions]
    
    def export_strategy(self, filepath: str) -> None:
        """Export complete strategy and decisions to file."""
        data = {
            "engine_id": str(self.engine_id),
            "strategy": self.strategy,
            "optimization_target": self.optimization_target,
            "config": self.config,
            "decisions": self.get_decision_log(),
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def create_fast_strategy(cls) -> ComputeAdaptiveScreeningEngine:
        """Factory method for fast screening strategy."""
        return cls(strategy="fast", optimization_target="speed")
    
    @classmethod
    def create_thorough_strategy(cls) -> ComputeAdaptiveScreeningEngine:
        """Factory method for thorough screening strategy."""
        return cls(strategy="thorough", optimization_target="enrichment")
    
    @classmethod
    def create_cost_optimized_strategy(cls) -> ComputeAdaptiveScreeningEngine:
        """Factory method for cost-optimized strategy."""
        return cls(strategy="balanced", optimization_target="cost")


# Convenience functions
def create_cas_engine(
    resources: ResourceDescriptor,
    target: TargetProfile,
    library: LibraryProfile,
    strategy: str = "balanced"
) -> ComputeAdaptiveScreeningEngine:
    """
    Convenience function to create CAS engine with profiles.
    
    Args:
        resources: Available compute resources
        target: Target profile
        library: Library profile
        strategy: Overall strategy
        
    Returns:
        Configured CAS engine
    """
    engine = ComputeAdaptiveScreeningEngine(strategy=strategy)
    return engine


def simulate_cas_decision(
    input_compounds: int,
    available_cpus: int,
    available_memory_gb: float,
    has_gpu: bool,
    target_difficulty: str = "moderate"
) -> Dict[str, Any]:
    """
    Simulate CAS engine decision making.
    
    Useful for testing and demonstration.
    """
    resources = ResourceDescriptor(
        total_cpu_cores=available_cpus,
        available_cpu_cores=available_cpus,
        total_memory_gb=available_memory_gb,
        available_memory_gb=available_memory_gb,
        gpu_available=has_gpu,
        gpu_count=1 if has_gpu else 0
    )
    
    target = TargetProfile(
        target_id=uuid4(),
        target_name="simulated_target",
        difficulty=TargetDifficulty(target_difficulty)
    )
    
    library = LibraryProfile(
        library_id="simulated_lib",
        library_name="Simulated Library",
        total_compounds=input_compounds
    )
    
    engine = ComputeAdaptiveScreeningEngine()
    
    # Simulate stage optimization
    stage_config = {
        "time_per_compound_seconds": 1.0,
        "min_cpu_cores": 1,
        "min_memory_gb": 4.0,
        "requires_gpu": False,
        "parallelizable": True
    }
    
    result = engine.optimize_stage_parameters(
        stage_id=uuid4(),
        stage_type="docking",
        input_compounds=input_compounds,
        resources=resources,
        target=target,
        library=library,
        current_config=stage_config
    )
    
    return {
        "optimized_threshold": result.filter_threshold,
        "top_percent": result.top_percent_selection,
        "predicted_output": result.predicted_output_count,
        "allocated_cpus": result.allocated_cpu_cores,
        "use_gpu": result.use_gpu,
        "predicted_time_hours": result.predicted_execution_time_seconds / 3600,
        "decisions_count": len(result.decisions)
    }
