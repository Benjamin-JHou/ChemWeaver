"""
CAS-VS Innovations Module
=========================

Mandatory Innovation Elements:
1. Screening Cost-Performance Frontier Modeling
2. Screening Uncertainty Propagation Framework

Author: CAS-VS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np


# ============================================================================
# INNOVATION 1: Screening Cost-Performance Frontier Modeling
# ============================================================================

@dataclass
class CostPerformancePoint:
    """
    Single point on the cost-performance frontier.
    
    Represents a specific configuration with its cost and expected performance.
    """
    configuration_id: UUID = field(default_factory=uuid4)
    
    # Configuration parameters
    stages_enabled: List[int] = field(default_factory=list)  # e.g., [0, 1, 3]
    filter_thresholds: Dict[int, float] = field(default_factory=dict)
    sampling_depth: str = "standard"  # minimal, standard, exhaustive
    ai_model_complexity: str = "standard"  # simple, standard, ensemble
    
    # Cost metrics
    estimated_cpu_hours: float = 0.0
    estimated_gpu_hours: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_time_seconds: float = 0.0
    
    # Performance predictions
    predicted_enrichment_factor: float = 1.0
    predicted_auc_roc: Optional[float] = None
    predicted_hits_at_1_percent: Optional[float] = None
    predicted_success_probability: float = 0.5
    
    # Quality metrics
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    
    def compute_value_score(self, value_weight: float = 1.0) -> float:
        """
        Compute value score balancing cost and performance.
        
        Value = Performance^value_weight / Cost
        """
        if self.estimated_cost_usd <= 0:
            return float('inf') if self.predicted_enrichment_factor > 1 else 0.0
        
        return (self.predicted_enrichment_factor ** value_weight) / self.estimated_cost_usd
    
    def dominates(self, other: CostPerformancePoint) -> bool:
        """
        Check if this point dominates another (Pareto dominance).
        
        Dominates if:
        - Has equal or better performance on all metrics
        - Has strictly better performance on at least one metric
        - Has equal or lower cost
        """
        better_performance = (
            self.predicted_enrichment_factor >= other.predicted_enrichment_factor and
            (self.predicted_auc_roc or 0) >= (other.predicted_auc_roc or 0) and
            (self.predicted_hits_at_1_percent or 0) >= (other.predicted_hits_at_1_percent or 0)
        )
        
        lower_cost = (
            self.estimated_cost_usd <= other.estimated_cost_usd and
            self.estimated_time_seconds <= other.estimated_time_seconds
        )
        
        strictly_better = (
            self.predicted_enrichment_factor > other.predicted_enrichment_factor or
            self.estimated_cost_usd < other.estimated_cost_usd
        )
        
        return better_performance and lower_cost and strictly_better
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration_id": str(self.configuration_id),
            "stages": self.stages_enabled,
            "thresholds": self.filter_thresholds,
            "sampling": self.sampling_depth,
            "ai_complexity": self.ai_model_complexity,
            "cost": {
                "cpu_hours": self.estimated_cpu_hours,
                "gpu_hours": self.estimated_gpu_hours,
                "usd": self.estimated_cost_usd,
                "time_seconds": self.estimated_time_seconds
            },
            "performance": {
                "enrichment": self.predicted_enrichment_factor,
                "auc_roc": self.predicted_auc_roc,
                "hits_at_1": self.predicted_hits_at_1_percent,
                "success_prob": self.predicted_success_probability
            },
            "confidence": [
                self.confidence_interval_lower,
                self.confidence_interval_upper
            ]
        }


class CostPerformanceFrontier:
    """
    INNOVATION 1: Screening Cost-Performance Frontier Modeling
    
    Models the tradeoff between screening cost and expected enrichment factor,
    enabling optimal resource allocation decisions.
    
    The frontier represents Pareto-optimal configurations where no other
    configuration provides better performance at lower cost.
    """
    
    def __init__(self, 
                 target_id: UUID,
                 library_id: str,
                 benchmark_data: Optional[Dict[str, Any]] = None):
        self.frontier_id = uuid4()
        self.target_id = target_id
        self.library_id = library_id
        self.benchmark_data = benchmark_data or {}
        
        # Frontier points
        self.points: List[CostPerformancePoint] = []
        self.pareto_optimal_points: List[CostPerformancePoint] = []
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.model_version = "1.0.0"
    
    def add_point(self, point: CostPerformancePoint) -> None:
        """Add a configuration point to the frontier."""
        self.points.append(point)
        self._update_pareto_frontier()
    
    def _update_pareto_frontier(self) -> None:
        """Recompute Pareto-optimal subset of points."""
        if not self.points:
            self.pareto_optimal_points = []
            return
        
        pareto = []
        for point in self.points:
            # Check if any other point dominates this one
            dominated = False
            for other in self.points:
                if other.configuration_id != point.configuration_id and other.dominates(point):
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(point)
        
        # Sort by cost for visualization
        self.pareto_optimal_points = sorted(pareto, key=lambda p: p.estimated_cost_usd)
    
    def find_optimal_configuration(
        self,
        budget_usd: Optional[float] = None,
        time_limit_seconds: Optional[float] = None,
        min_enrichment: float = 5.0,
        optimization_target: str = "balanced"  # performance, cost, balanced
    ) -> Optional[CostPerformancePoint]:
        """
        Find optimal configuration given constraints.
        
        Args:
            budget_usd: Maximum budget constraint
            time_limit_seconds: Maximum time constraint
            min_enrichment: Minimum required enrichment factor
            optimization_target: Optimization objective
            
        Returns:
            Best configuration satisfying constraints, or None if infeasible
        """
        feasible = self.pareto_optimal_points.copy()
        
        # Apply budget constraint
        if budget_usd is not None:
            feasible = [p for p in feasible if p.estimated_cost_usd <= budget_usd]
        
        # Apply time constraint
        if time_limit_seconds is not None:
            feasible = [p for p in feasible if p.estimated_time_seconds <= time_limit_seconds]
        
        # Apply enrichment constraint
        feasible = [p for p in feasible if p.predicted_enrichment_factor >= min_enrichment]
        
        if not feasible:
            return None
        
        # Select based on optimization target
        if optimization_target == "performance":
            # Maximize enrichment
            return max(feasible, key=lambda p: p.predicted_enrichment_factor)
        elif optimization_target == "cost":
            # Minimize cost
            return min(feasible, key=lambda p: p.estimated_cost_usd)
        else:  # balanced
            # Maximize value score
            return max(feasible, key=lambda p: p.compute_value_score(value_weight=1.0))
    
    def compute_frontier_efficiency(
        self,
        point: CostPerformancePoint
    ) -> float:
        """
        Compute efficiency of a point relative to the frontier.
        
        Efficiency = (EF - EF_min) / (EF_max - EF_min) at same cost
        """
        if not self.pareto_optimal_points:
            return 0.0
        
        # Find points with similar cost
        cost_range = point.estimated_cost_usd * 0.1  # ±10%
        similar_cost = [
            p for p in self.pareto_optimal_points
            if abs(p.estimated_cost_usd - point.estimated_cost_usd) <= cost_range
        ]
        
        if not similar_cost:
            return 0.5  # Neutral if no comparison points
        
        ef_values = [p.predicted_enrichment_factor for p in similar_cost]
        ef_min, ef_max = min(ef_values), max(ef_values)
        
        if ef_max == ef_min:
            return 1.0 if point.predicted_enrichment_factor >= ef_max else 0.0
        
        efficiency = (point.predicted_enrichment_factor - ef_min) / (ef_max - ef_min)
        return max(0.0, min(1.0, efficiency))
    
    def interpolate_performance(
        self,
        cost_usd: float
    ) -> Dict[str, float]:
        """
        Interpolate expected performance at a given cost.
        
        Uses linear interpolation between frontier points.
        """
        if not self.pareto_optimal_points:
            return {"enrichment": 1.0, "confidence": 0.0}
        
        # Sort by cost
        sorted_points = sorted(self.pareto_optimal_points, key=lambda p: p.estimated_cost_usd)
        
        # Find surrounding points
        lower = None
        upper = None
        
        for point in sorted_points:
            if point.estimated_cost_usd <= cost_usd:
                lower = point
            if point.estimated_cost_usd >= cost_usd and upper is None:
                upper = point
                break
        
        # Handle edge cases
        if lower is None:
            return {"enrichment": sorted_points[0].predicted_enrichment_factor, "confidence": 0.5}
        if upper is None or upper == lower:
            return {"enrichment": lower.predicted_enrichment_factor, "confidence": 0.8}
        
        # Linear interpolation
        cost_range = upper.estimated_cost_usd - lower.estimated_cost_usd
        if cost_range == 0:
            return {"enrichment": lower.predicted_enrichment_factor, "confidence": 0.8}
        
        t = (cost_usd - lower.estimated_cost_usd) / cost_range
        interpolated_ef = (
            lower.predicted_enrichment_factor * (1 - t) +
            upper.predicted_enrichment_factor * t
        )
        
        return {
            "enrichment": interpolated_ef,
            "confidence": 0.8 - 0.3 * abs(0.5 - t)  # Higher confidence near known points
        }
    
    def generate_recommendation_report(self) -> str:
        """Generate human-readable frontier analysis report."""
        lines = [
            "=" * 70,
            "COST-PERFORMANCE FRONTIER ANALYSIS",
            "=" * 70,
            "",
            f"Frontier ID: {self.frontier_id}",
            f"Target: {self.target_id}",
            f"Library: {self.library_id}",
            f"Points: {len(self.points)} ({len(self.pareto_optimal_points)} Pareto-optimal)",
            "",
            "PARETO-OPTIMAL CONFIGURATIONS",
            "-" * 70,
        ]
        
        for i, point in enumerate(self.pareto_optimal_points, 1):
            lines.append(f"\n{i}. Configuration {point.configuration_id.hex[:8]}...")
            lines.append(f"   Stages: {point.stages_enabled}")
            lines.append(f"   Cost: ${point.estimated_cost_usd:.2f} ({point.estimated_time_seconds/3600:.1f}h)")
            lines.append(f"   Expected EF: {point.predicted_enrichment_factor:.2f}")
            lines.append(f"   Success Prob: {point.predicted_success_probability:.1%}")
            lines.append(f"   Value Score: {point.compute_value_score():.4f}")
        
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 70,
        ])
        
        # Budget-constrained recommendation
        budget_optimal = self.find_optimal_configuration(budget_usd=1000)
        if budget_optimal:
            lines.append(f"\nBudget-constrained ($1000):")
            lines.append(f"  → Configuration {budget_optimal.configuration_id.hex[:8]}...")
            lines.append(f"  → Expected EF: {budget_optimal.predicted_enrichment_factor:.2f}")
        
        # Time-constrained recommendation
        time_optimal = self.find_optimal_configuration(time_limit_seconds=86400)
        if time_optimal:
            lines.append(f"\nTime-constrained (24h):")
            lines.append(f"  → Configuration {time_optimal.configuration_id.hex[:8]}...")
            lines.append(f"  → Expected EF: {time_optimal.predicted_enrichment_factor:.2f}")
        
        lines.extend([
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier_id": str(self.frontier_id),
            "target_id": str(self.target_id),
            "library_id": self.library_id,
            "created_at": self.created_at.isoformat(),
            "model_version": self.model_version,
            "points": [p.to_dict() for p in self.points],
            "pareto_points": [p.to_dict() for p in self.pareto_optimal_points]
        }


# ============================================================================
# INNOVATION 2: Screening Uncertainty Propagation Framework
# ============================================================================

@dataclass
class UncertaintyComponent:
    """Individual uncertainty component."""
    source: str  # aleatoric, epistemic, model, data
    magnitude: float  # Standard deviation or entropy
    confidence_level: float = 0.95
    
    def get_confidence_interval(self, value: float) -> Tuple[float, float]:
        """Get confidence interval around a value."""
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z_score * self.magnitude
        return (value - margin, value + margin)


@dataclass
class StageUncertainty:
    """Uncertainty estimate for a single screening stage."""
    stage_id: UUID
    stage_type: str
    
    # Uncertainty components
    components: List[UncertaintyComponent] = field(default_factory=list)
    
    # Aggregated uncertainty
    total_uncertainty: float = 0.0
    propagated_from_previous: float = 0.0
    
    # Predictions
    predicted_score: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    prediction_entropy: Optional[float] = None
    
    def aggregate_uncertainty(self) -> float:
        """
        Aggregate uncertainty components.
        
        Uses sum of variances (square root for standard deviation).
        """
        total_variance = sum(comp.magnitude ** 2 for comp in self.components)
        self.total_uncertainty = np.sqrt(total_variance + self.propagated_from_previous ** 2)
        return self.total_uncertainty
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": str(self.stage_id),
            "stage_type": self.stage_type,
            "components": [
                {"source": c.source, "magnitude": c.magnitude}
                for c in self.components
            ],
            "total_uncertainty": self.total_uncertainty,
            "predicted_score": self.predicted_score,
            "confidence_interval": self.confidence_interval
        }


class UncertaintyPropagationFramework:
    """
    INNOVATION 2: Screening Uncertainty Propagation Framework
    
    Propagates uncertainty across screening stages to provide
    calibrated confidence estimates at each decision point.
    
    Models:
    - Aleatoric uncertainty (inherent data noise)
    - Epistemic uncertainty (model uncertainty)
    - Propagation through nonlinear transformations
    - Stage-specific uncertainty accumulation
    """
    
    def __init__(self, pipeline_id: UUID):
        self.framework_id = uuid4()
        self.pipeline_id = pipeline_id
        
        # Uncertainty tracking per stage
        self.stage_uncertainties: Dict[UUID, StageUncertainty] = {}
        
        # Propagation rules
        self.propagation_functions: Dict[str, callable] = {
            "linear": self._linear_propagation,
            "multiplicative": self._multiplicative_propagation,
            "threshold": self._threshold_propagation
        }
    
    def register_stage(
        self,
        stage_id: UUID,
        stage_type: str,
        initial_uncertainty: List[UncertaintyComponent]
    ) -> StageUncertainty:
        """Register a stage and initialize uncertainty tracking."""
        stage_unc = StageUncertainty(
            stage_id=stage_id,
            stage_type=stage_type,
            components=initial_uncertainty
        )
        stage_unc.aggregate_uncertainty()
        
        self.stage_uncertainties[stage_id] = stage_unc
        return stage_unc
    
    def propagate_uncertainty(
        self,
        from_stage_id: UUID,
        to_stage_id: UUID,
        propagation_type: str = "linear"
    ) -> StageUncertainty:
        """
        Propagate uncertainty from one stage to the next.
        
        Args:
            from_stage_id: Source stage
            to_stage_id: Target stage
            propagation_type: How uncertainty propagates (linear, multiplicative, threshold)
            
        Returns:
            Updated uncertainty for target stage
        """
        if from_stage_id not in self.stage_uncertainties:
            raise ValueError(f"Source stage {from_stage_id} not registered")
        
        if to_stage_id not in self.stage_uncertainties:
            raise ValueError(f"Target stage {to_stage_id} not registered")
        
        source_unc = self.stage_uncertainties[from_stage_id]
        target_unc = self.stage_uncertainties[to_stage_id]
        
        # Apply propagation function
        prop_func = self.propagation_functions.get(
            propagation_type, 
            self._linear_propagation
        )
        propagated = prop_func(source_unc.total_uncertainty)
        
        # Update target stage
        target_unc.propagated_from_previous = propagated
        target_unc.aggregate_uncertainty()
        
        return target_unc
    
    def _linear_propagation(self, uncertainty: float) -> float:
        """Linear uncertainty propagation (additive)."""
        return uncertainty
    
    def _multiplicative_propagation(self, uncertainty: float) -> float:
        """Multiplicative uncertainty propagation (scales with stage complexity)."""
        return uncertainty * 1.2  # 20% amplification per stage
    
    def _threshold_propagation(self, uncertainty: float) -> float:
        """Threshold-based propagation (uncertainty decreases after filtering)."""
        # Filtering typically reduces uncertainty
        return uncertainty * 0.8
    
    def compute_compound_uncertainty(
        self,
        compound_id: UUID,
        stage_id: UUID
    ) -> Dict[str, Any]:
        """
        Compute total uncertainty for a specific compound at a stage.
        
        Considers both stage-level and compound-level uncertainty.
        """
        if stage_id not in self.stage_uncertainties:
            return {"error": "Stage not registered"}
        
        stage_unc = self.stage_uncertainties[stage_id]
        
        # In real implementation, would lookup compound-specific uncertainty
        # For now, return stage-level uncertainty
        return {
            "compound_id": str(compound_id),
            "stage_id": str(stage_id),
            "total_uncertainty": stage_unc.total_uncertainty,
            "confidence_interval": stage_unc.confidence_interval,
            "reliability": self._compute_reliability(stage_unc.total_uncertainty)
        }
    
    def _compute_reliability(self, uncertainty: float) -> str:
        """Map uncertainty to reliability category."""
        if uncertainty < 0.1:
            return "high"
        elif uncertainty < 0.3:
            return "medium"
        else:
            return "low"
    
    def get_uncertainty_budget(self) -> Dict[str, float]:
        """
        Get remaining uncertainty budget across pipeline.
        
        Useful for deciding when to stop or request more data.
        """
        total_uncertainty = sum(
            s.total_uncertainty for s in self.stage_uncertainties.values()
        )
        
        return {
            "total_pipeline_uncertainty": total_uncertainty,
            "average_stage_uncertainty": total_uncertainty / len(self.stage_uncertainties) if self.stage_uncertainties else 0,
            "max_stage_uncertainty": max(s.total_uncertainty for s in self.stage_uncertainties.values()) if self.stage_uncertainties else 0
        }
    
    def generate_uncertainty_report(self) -> str:
        """Generate comprehensive uncertainty propagation report."""
        lines = [
            "=" * 70,
            "UNCERTAINTY PROPAGATION FRAMEWORK REPORT",
            "=" * 70,
            "",
            f"Framework ID: {self.framework_id}",
            f"Pipeline: {self.pipeline_id}",
            f"Stages Tracked: {len(self.stage_uncertainties)}",
            "",
            "STAGE-LEVEL UNCERTAINTY",
            "-" * 70,
        ]
        
        for stage_id, unc in self.stage_uncertainties.items():
            lines.append(f"\nStage {stage_id.hex[:8]}... ({unc.stage_type})")
            lines.append(f"  Total Uncertainty: {unc.total_uncertainty:.4f}")
            lines.append(f"  Propagated: {unc.propagated_from_previous:.4f}")
            lines.append(f"  Components:")
            for comp in unc.components:
                lines.append(f"    - {comp.source}: {comp.magnitude:.4f}")
        
        budget = self.get_uncertainty_budget()
        lines.extend([
            "",
            "UNCERTAINTY BUDGET",
            "-" * 70,
            f"Total Pipeline: {budget['total_pipeline_uncertainty']:.4f}",
            f"Average per Stage: {budget['average_stage_uncertainty']:.4f}",
            f"Maximum Stage: {budget['max_stage_uncertainty']:.4f}",
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework_id": str(self.framework_id),
            "pipeline_id": str(self.pipeline_id),
            "stages": {
                str(sid): unc.to_dict()
                for sid, unc in self.stage_uncertainties.items()
            }
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_cost_performance_frontier(
    target_id: UUID,
    library_id: str,
    sample_configurations: int = 50
) -> CostPerformanceFrontier:
    """
    Create a cost-performance frontier with sample configurations.
    
    Useful for demonstration and testing.
    """
    frontier = CostPerformanceFrontier(target_id, library_id)
    
    # Generate sample configurations
    import random
    random.seed(42)
    
    for i in range(sample_configurations):
        # Random stage selection
        stages = random.sample([0, 1, 2, 3, 4], random.randint(2, 5))
        stages.sort()
        
        # Random cost
        cost = 50 + random.random() * 950  # $50-$1000
        
        # Performance correlates with cost and stage depth
        base_ef = 2.0 + len(stages) * 1.5
        cost_bonus = (cost / 1000) * 3.0
        enrichment = base_ef + cost_bonus + random.gauss(0, 0.5)
        enrichment = max(1.0, enrichment)
        
        point = CostPerformancePoint(
            stages_enabled=stages,
            estimated_cost_usd=cost,
            estimated_time_seconds=cost * 10,  # 10s per dollar
            predicted_enrichment_factor=enrichment,
            predicted_success_probability=min(0.95, 0.3 + len(stages) * 0.15)
        )
        
        frontier.add_point(point)
    
    return frontier


def create_uncertainty_framework(
    pipeline_id: UUID,
    stage_ids: List[UUID],
    stage_types: List[str]
) -> UncertaintyPropagationFramework:
    """
    Create uncertainty framework with typical uncertainty levels.
    """
    framework = UncertaintyPropagationFramework(pipeline_id)
    
    # Register stages with typical uncertainties
    for stage_id, stage_type in zip(stage_ids, stage_types):
        components = []
        
        if stage_type == "docking":
            components = [
                UncertaintyComponent("model", 0.3),
                UncertaintyComponent("data", 0.2)
            ]
        elif stage_type == "ai":
            components = [
                UncertaintyComponent("model", 0.25),
                UncertaintyComponent("epistemic", 0.15)
            ]
        else:
            components = [
                UncertaintyComponent("data", 0.15)
            ]
        
        framework.register_stage(stage_id, stage_type, components)
    
    return framework


# Export
__all__ = [
    'CostPerformancePoint',
    'CostPerformanceFrontier',
    'UncertaintyComponent',
    'StageUncertainty',
    'UncertaintyPropagationFramework',
    'create_cost_performance_frontier',
    'create_uncertainty_framework'
]
