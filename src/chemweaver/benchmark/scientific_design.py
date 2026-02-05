"""
VS-Bench - Virtual Screening Benchmark System
Scientific Design and Architecture
===============================================

Comprehensive benchmark ecosystem for AI-assisted virtual screening
covering docking surrogates, ADMET, multi-target activity, and OOD generalization.

Author: VS-Bench Development Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
import json


class CapabilityAxis(Enum):
    """
    Benchmark capability axes defining evaluation dimensions.
    
    Each axis represents a distinct capability that AI models
    should demonstrate for virtual screening applications.
    """
    DOCKING_SURROGATE = "docking_surrogate"
    MULTI_TARGET_ACTIVITY = "multi_target_activity"
    ADMET_PREDICTION = "admet_prediction"
    OOD_GENERALIZATION = "ood_generalization"
    CROSS_PROTEIN_TRANSFER = "cross_protein_transfer"
    EXPERIMENTAL_VALIDATION = "experimental_validation"


class BenchmarkSplit(Enum):
    """Types of data splits for robust evaluation."""
    SCAFFOLD = "scaffold"  # Bemis-Murcko scaffold split
    TIME = "time"  # Temporal split
    PROTEIN_FAMILY = "protein_family"  # Family-based split
    CHEMICAL_OOD = "chemical_ood"  # Chemical space OOD
    RANDOM = "random"  # Random stratified split


@dataclass
class BenchmarkCapability:
    """
    Definition of a benchmark capability axis.
    
    Defines what a model is evaluated on, including
    metrics, datasets, and success criteria.
    """
    axis: CapabilityAxis
    name: str
    description: str
    
    # Evaluation configuration
    primary_metrics: List[str] = field(default_factory=list)
    secondary_metrics: List[str] = field(default_factory=list)
    
    # Dataset requirements
    min_compounds: int = 1000
    min_targets: int = 1
    
    # Success criteria
    baseline_score: float = 0.5  # Random baseline
    target_score: float = 0.8     # State-of-the-art target
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis.value,
            "name": self.name,
            "description": self.description,
            "metrics": {
                "primary": self.primary_metrics,
                "secondary": self.secondary_metrics
            },
            "requirements": {
                "min_compounds": self.min_compounds,
                "min_targets": self.min_targets
            },
            "criteria": {
                "baseline": self.baseline_score,
                "target": self.target_score
            }
        }


@dataclass
class SubBenchmark:
    """
    Individual sub-benchmark within the suite.
    
    A sub-benchmark focuses on a specific task or dataset
    with defined train/val/test splits and evaluation protocol.
    """
    benchmark_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    
    # Capability alignment
    capability_axes: List[CapabilityAxis] = field(default_factory=list)
    
    # Dataset
    dataset_name: str = ""
    dataset_version: str = "1.0.0"
    
    # Data splits
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    split_strategy: BenchmarkSplit = BenchmarkSplit.SCAFFOLD
    
    # Task type
    task_type: str = "regression"  # regression, classification, ranking
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    data_card: Optional[str] = None
    
    def get_info(self) -> Dict[str, Any]:
        """Get benchmark information summary."""
        return {
            "id": str(self.benchmark_id),
            "name": self.name,
            "description": self.description,
            "capabilities": [axis.value for axis in self.capability_axes],
            "dataset": {
                "name": self.dataset_name,
                "version": self.dataset_version
            },
            "splits": {
                "train": self.train_size,
                "val": self.val_size,
                "test": self.test_size,
                "strategy": self.split_strategy.value
            },
            "task": self.task_type
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for virtual screening.
    
    Aggregates multiple sub-benchmarks covering different
    capability axes and evaluation scenarios.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.suite_id = uuid4()
        self.name = name
        self.version = version
        self.created_at = datetime.utcnow()
        
        # Benchmark components
        self.capabilities: Dict[CapabilityAxis, BenchmarkCapability] = {}
        self.subbenchmarks: Dict[str, SubBenchmark] = {}
        
        # Configuration
        self.global_weights: Dict[str, float] = {}
        
        self._initialize_default_capabilities()
        self._initialize_default_weights()
    
    def _initialize_default_capabilities(self) -> None:
        """Initialize default capability axes."""
        self.capabilities[CapabilityAxis.DOCKING_SURROGATE] = BenchmarkCapability(
            axis=CapabilityAxis.DOCKING_SURROGATE,
            name="Docking Surrogate Prediction",
            description="Predict molecular docking scores as a surrogate for expensive docking calculations",
            primary_metrics=["rmse", "spearman", "concordance_index"],
            secondary_metrics=["mae", "r2", "top_k_recall"],
            min_compounds=100000,
            min_targets=10,
            baseline_score=0.3,
            target_score=0.85
        )
        
        self.capabilities[CapabilityAxis.MULTI_TARGET_ACTIVITY] = BenchmarkCapability(
            axis=CapabilityAxis.MULTI_TARGET_ACTIVITY,
            name="Multi-Target Activity Prediction",
            description="Simultaneously predict activity against multiple protein targets",
            primary_metrics=["auc_roc", "bedroc", "enrichment_factor_1"],
            secondary_metrics=["pr_auc", "f1", "mcc"],
            min_compounds=50000,
            min_targets=20,
            baseline_score=0.5,
            target_score=0.9
        )
        
        self.capabilities[CapabilityAxis.ADMET_PREDICTION] = BenchmarkCapability(
            axis=CapabilityAxis.ADMET_PREDICTION,
            name="ADMET Multi-Task Prediction",
            description="Predict absorption, distribution, metabolism, excretion, and toxicity properties",
            primary_metrics=["rmse", "mae", "classification_accuracy"],
            secondary_metrics=["r2", "balanced_accuracy", "kappa"],
            min_compounds=20000,
            min_targets=5,  # 5 ADMET endpoints
            baseline_score=0.4,
            target_score=0.8
        )
        
        self.capabilities[CapabilityAxis.OOD_GENERALIZATION] = BenchmarkCapability(
            axis=CapabilityAxis.OOD_GENERALIZATION,
            name="OOD Chemical Space Generalization",
            description="Generalize to chemically distinct scaffolds and structural motifs",
            primary_metrics=["ood_performance_ratio", "scaffold_test_auc"],
            secondary_metrics=["domain_gap", "transfer_efficiency"],
            min_compounds=50000,
            min_targets=5,
            baseline_score=0.2,
            target_score=0.75
        )
        
        self.capabilities[CapabilityAxis.CROSS_PROTEIN_TRANSFER] = BenchmarkCapability(
            axis=CapabilityAxis.CROSS_PROTEIN_TRANSFER,
            name="Cross-Protein Family Transfer",
            description="Transfer knowledge across different protein families and target classes",
            primary_metrics=["cross_family_auc", "transfer_score"],
            secondary_metrics=["family_similarity", "negative_transfer"],
            min_compounds=50000,
            min_targets=30,
            baseline_score=0.25,
            target_score=0.7
        )
        
        self.capabilities[CapabilityAxis.EXPERIMENTAL_VALIDATION] = BenchmarkCapability(
            axis=CapabilityAxis.EXPERIMENTAL_VALIDATION,
            name="Experimental Subset Validation",
            description="Validate predictions on experimentally measured holdout set",
            primary_metrics=["experimental_correlation", "hit_rate"],
            secondary_metrics=["enrichment", "false_positive_rate"],
            min_compounds=1000,
            min_targets=1,
            baseline_score=0.1,
            target_score=0.6
        )
    
    def _initialize_default_weights(self) -> None:
        """Initialize default global scoring weights."""
        self.global_weights = {
            "docking_surrogate": 0.25,
            "admet_prediction": 0.25,
            "multi_target_activity": 0.20,
            "ood_generalization": 0.15,
            "cross_protein_transfer": 0.15
        }
    
    def add_subbenchmark(self, benchmark: SubBenchmark) -> None:
        """Add a sub-benchmark to the suite."""
        self.subbenchmarks[benchmark.name] = benchmark
    
    def get_benchmarks_by_capability(
        self,
        capability: CapabilityAxis
    ) -> List[SubBenchmark]:
        """Get all benchmarks testing a specific capability."""
        return [
            b for b in self.subbenchmarks.values()
            if capability in b.capability_axes
        ]
    
    def compute_global_score(
        self,
        benchmark_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute weighted global score across all benchmarks.
        
        Args:
            benchmark_scores: Dict mapping benchmark name to score
            
        Returns:
            Global score and component breakdown
        """
        # Group by capability
        capability_scores: Dict[CapabilityAxis, List[float]] = {
            axis: [] for axis in CapabilityAxis
        }
        
        for bench_name, score in benchmark_scores.items():
            if bench_name in self.subbenchmarks:
                bench = self.subbenchmarks[bench_name]
                for axis in bench.capability_axes:
                    capability_scores[axis].append(score)
        
        # Average within each capability
        capability_averages = {}
        for axis, scores in capability_scores.items():
            if scores:
                capability_averages[axis.value] = sum(scores) / len(scores)
            else:
                capability_averages[axis.value] = 0.0
        
        # Weighted global score
        global_score = 0.0
        for axis_name, weight in self.global_weights.items():
            axis_enum = CapabilityAxis(axis_name)
            if axis_enum in [CapabilityAxis.EXPERIMENTAL_VALIDATION]:
                continue  # Not included in global score
            axis_value = capability_averages.get(axis_name, 0.0)
            global_score += axis_value * weight
        
        return {
            "global_score": global_score,
            "by_capability": capability_averages,
            "by_benchmark": benchmark_scores
        }
    
    def get_suite_info(self) -> Dict[str, Any]:
        """Get complete suite information."""
        return {
            "suite_id": str(self.suite_id),
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "capabilities": {
                axis.value: cap.to_dict()
                for axis, cap in self.capabilities.items()
            },
            "subbenchmarks": [
                bench.get_info() for bench in self.subbenchmarks.values()
            ],
            "weights": self.global_weights
        }


# Predefined benchmark suites

def create_standard_suite() -> BenchmarkSuite:
    """Create the standard VS-Bench suite."""
    suite = BenchmarkSuite(
        name="VS-Bench Standard",
        version="1.0.0"
    )
    
    # Docking Surrogate Benchmark
    suite.add_subbenchmark(SubBenchmark(
        name="docking_surrogate_main",
        description="Main docking score prediction benchmark on diverse protein targets",
        capability_axes=[CapabilityAxis.DOCKING_SURROGATE],
        dataset_name="pdbbind_docked",
        dataset_version="2020_v1",
        train_size=150000,
        val_size=15000,
        test_size=15000,
        split_strategy=BenchmarkSplit.SCAFFOLD,
        task_type="regression"
    ))
    
    suite.add_subbenchmark(SubBenchmark(
        name="docking_surrogate_ood",
        description="Docking prediction on out-of-distribution scaffolds",
        capability_axes=[CapabilityAxis.DOCKING_SURROGATE, CapabilityAxis.OOD_GENERALIZATION],
        dataset_name="pdbbind_docked_ood",
        dataset_version="2020_v1",
        train_size=100000,
        val_size=10000,
        test_size=20000,
        split_strategy=BenchmarkSplit.CHEMICAL_OOD,
        task_type="regression"
    ))
    
    # ADMET Benchmark
    suite.add_subbenchmark(SubBenchmark(
        name="admet_multi_task",
        description="Multi-task ADMET property prediction",
        capability_axes=[CapabilityAxis.ADMET_PREDICTION],
        dataset_name="admet_collection",
        dataset_version="2023_v1",
        train_size=80000,
        val_size=8000,
        test_size=8000,
        split_strategy=BenchmarkSplit.SCAFFOLD,
        task_type="multitask"
    ))
    
    # Multi-Target Activity Benchmark
    suite.add_subbenchmark(SubBenchmark(
        name="multi_target_activity",
        description="Activity prediction across 50 protein targets",
        capability_axes=[CapabilityAxis.MULTI_TARGET_ACTIVITY],
        dataset_name="chembl_multi_target",
        dataset_version="33_v1",
        train_size=250000,
        val_size=25000,
        test_size=25000,
        split_strategy=BenchmarkSplit.TIME,
        task_type="multilabel"
    ))
    
    # OOD Generalization Benchmark
    suite.add_subbenchmark(SubBenchmark(
        name="ood_generalization",
        description="Generalization to novel chemical scaffolds",
        capability_axes=[CapabilityAxis.OOD_GENERALIZATION],
        dataset_name="chembl_scaffold_ood",
        dataset_version="33_v1",
        train_size=180000,
        val_size=10000,
        test_size=20000,
        split_strategy=BenchmarkSplit.SCAFFOLD,
        task_type="classification"
    ))
    
    # Cross-Protein Transfer Benchmark
    suite.add_subbenchmark(SubBenchmark(
        name="cross_protein_transfer",
        description="Transfer across kinase, GPCR, and protease families",
        capability_axes=[CapabilityAxis.CROSS_PROTEIN_TRANSFER],
        dataset_name="chembl_cross_family",
        dataset_version="33_v1",
        train_size=200000,
        val_size=20000,
        test_size=30000,
        split_strategy=BenchmarkSplit.PROTEIN_FAMILY,
        task_type="multilabel"
    ))
    
    # Experimental Validation Benchmark
    suite.add_subbenchmark(SubBenchmark(
        name="experimental_validation",
        description="Holdout set with experimental binding measurements",
        capability_axes=[CapabilityAxis.EXPERIMENTAL_VALIDATION],
        dataset_name="experimental_holdout",
        dataset_version="2024_v1",
        train_size=0,  # No training data provided
        val_size=0,
        test_size=5000,
        split_strategy=BenchmarkSplit.RANDOM,
        task_type="regression"
    ))
    
    return suite


def create_minimal_suite() -> BenchmarkSuite:
    """Create minimal benchmark suite for quick testing."""
    suite = BenchmarkSuite(
        name="VS-Bench Minimal",
        version="1.0.0-mini"
    )
    
    suite.add_subbenchmark(SubBenchmark(
        name="mini_docking",
        description="Small docking benchmark for development",
        capability_axes=[CapabilityAxis.DOCKING_SURROGATE],
        dataset_name="mini_docking",
        train_size=10000,
        val_size=1000,
        test_size=1000,
        split_strategy=BenchmarkSplit.RANDOM,
        task_type="regression"
    ))
    
    return suite


# Export
__all__ = [
    'CapabilityAxis',
    'BenchmarkSplit',
    'BenchmarkCapability',
    'SubBenchmark',
    'BenchmarkSuite',
    'create_standard_suite',
    'create_minimal_suite'
]
