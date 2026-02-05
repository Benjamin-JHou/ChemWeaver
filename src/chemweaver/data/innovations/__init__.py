"""
Virtual Screening Standard Schema (VSSS) - Novel Innovations
=============================================================

Two mandatory novel innovations:
1. Docking Reproducibility Hash - Deterministic fingerprinting of docking experiments
2. AI Confidence Calibration Metadata - Calibration and domain applicability tracking

Author: VSSS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from ..core.entities import DockingExperimentMetadata, DockingEngine, ScoringFunction


@dataclass
class DockingReproducibilityHash:
    """
    INNOVATION 1: Docking Reproducibility Hash
    
    A deterministic fingerprint encoding of all parameters that affect
    docking experiment outcomes. This hash ensures bit-for-bit reproducibility
    of docking experiments across different execution environments.
    
    The hash captures:
    - Docking engine identity and version
    - All parameters (even defaults)
    - Grid definition and binding site
    - Structure version and target identity
    - Random seed
    - Container/environment specification
    
    Purpose: Ensure identical docking experiment reproducibility across
    heterogeneous compute infrastructure.
    """
    
    # Hash components
    engine_component: str
    parameters_component: str
    grid_component: str
    structure_component: str
    environment_component: str
    
    # Final hash
    reproducibility_hash: str
    hash_algorithm: str = "sha256"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    schema_version: str = "1.0.0"
    
    @classmethod
    def from_experiment(
        cls,
        experiment: DockingExperimentMetadata,
        target_sequence_hash: Optional[str] = None,
        target_structure_hash: Optional[str] = None,
        grid_definition_hash: Optional[str] = None
    ) -> "DockingReproducibilityHash":
        """
        Generate reproducibility hash from experiment metadata.
        
        This method creates a deterministic fingerprint that uniquely
        identifies a docking experiment configuration.
        """
        # Component 1: Engine identity
        engine_data = {
            "engine": experiment.docking_engine.value,
            "engine_version": experiment.engine_version,
            "engine_git_commit": experiment.engine_git_commit,
            "scoring_function": experiment.scoring_function.value,
            "scoring_function_version": experiment.scoring_function_version
        }
        engine_component = cls._hash_dict(engine_data)
        
        # Component 2: Parameters (normalized to ensure consistency)
        params_data = {
            "exhaustiveness": experiment.parameters.exhaustiveness,
            "num_modes": experiment.parameters.num_modes,
            "energy_range": experiment.parameters.energy_range,
            "cpu_count": experiment.parameters.cpu_count,
            "seed": experiment.parameters.seed,
            "custom_parameters": cls._sort_dict(experiment.parameters.custom_parameters)
        }
        parameters_component = cls._hash_dict(params_data)
        
        # Component 3: Grid definition
        grid_data = {
            "protocol_name": experiment.grid_protocol.protocol_name,
            "receptor_preparation": experiment.grid_protocol.receptor_preparation,
            "grid_generation_tool": experiment.grid_protocol.grid_generation_tool,
            "padding": experiment.grid_protocol.padding,
            "spacing": experiment.grid_protocol.spacing,
            "grid_definition_hash": grid_definition_hash or "unknown"
        }
        grid_component = cls._hash_dict(grid_data)
        
        # Component 4: Structure information
        structure_data = {
            "target_id": str(experiment.target_id),
            "target_sequence_hash": target_sequence_hash or "unknown",
            "target_structure_hash": target_structure_hash or "unknown",
            "compound_library_id": experiment.compound_library_id
        }
        structure_component = cls._hash_dict(structure_data)
        
        # Component 5: Execution environment
        env_data = {
            "container_hash": experiment.execution_container_hash,
            "workflow_version": experiment.workflow_definition_version,
            "parameter_template_version": experiment.parameter_template_version,
            "execution_backend": experiment.execution_backend
        }
        environment_component = cls._hash_dict(env_data)
        
        # Final composite hash
        composite_data = {
            "engine": engine_component,
            "parameters": parameters_component,
            "grid": grid_component,
            "structure": structure_component,
            "environment": environment_component
        }
        reproducibility_hash = cls._hash_dict(composite_data)
        
        return cls(
            engine_component=engine_component,
            parameters_component=parameters_component,
            grid_component=grid_component,
            structure_component=structure_component,
            environment_component=environment_component,
            reproducibility_hash=reproducibility_hash
        )
    
    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        """Create deterministic hash from dictionary."""
        # Serialize with sorted keys for consistency
        serialized = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode()).hexdigest()[:32]
    
    @staticmethod
    def _sort_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sort dictionary for consistent serialization."""
        sorted_dict = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                sorted_dict[key] = DockingReproducibilityHash._sort_dict(value)
            elif isinstance(value, list):
                sorted_dict[key] = sorted(value) if all(isinstance(x, str) for x in value) else value
            else:
                sorted_dict[key] = value
        return sorted_dict
    
    def verify_compatibility(
        self,
        other: "DockingReproducibilityHash",
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Verify compatibility with another reproducibility hash.
        
        Args:
            other: Another reproducibility hash to compare
            strict: If True, requires exact match; if False, allows
                   engine version differences within minor versions
        
        Returns:
            (is_compatible, list_of_differences)
        """
        differences = []
        
        if strict:
            if self.reproducibility_hash != other.reproducibility_hash:
                differences.append("Full reproducibility hash mismatch")
        
        # Component-level comparison
        if self.engine_component != other.engine_component:
            differences.append("Engine configuration differs")
        
        if self.parameters_component != other.parameters_component:
            differences.append("Parameters differ")
        
        if self.grid_component != other.grid_component:
            differences.append("Grid definition differs")
        
        if self.structure_component != other.structure_component:
            differences.append("Structure differs")
        
        if self.environment_component != other.environment_component:
            differences.append("Environment differs")
        
        return len(differences) == 0, differences
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reproducibility_hash": self.reproducibility_hash,
            "hash_algorithm": self.hash_algorithm,
            "components": {
                "engine": self.engine_component,
                "parameters": self.parameters_component,
                "grid": self.grid_component,
                "structure": self.structure_component,
                "environment": self.environment_component
            },
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version
        }


@dataclass
class ConfidenceBin:
    """
    Single bin for confidence calibration.
    
    Represents a range of predicted probabilities and tracks
    the actual accuracy within that range.
    """
    bin_lower: float  # Lower bound (inclusive)
    bin_upper: float  # Upper bound (exclusive)
    
    # Statistics
    total_samples: int = 0
    positive_samples: int = 0
    
    @property
    def bin_center(self) -> float:
        """Center of the bin."""
        return (self.bin_lower + self.bin_upper) / 2
    
    @property
    def observed_accuracy(self) -> Optional[float]:
        """Observed accuracy in this bin."""
        if self.total_samples == 0:
            return None
        return self.positive_samples / self.total_samples
    
    @property
    def expected_accuracy(self) -> float:
        """Expected accuracy (bin center)."""
        return self.bin_center
    
    def calibration_error(self) -> Optional[float]:
        """Calibration error for this bin."""
        obs = self.observed_accuracy
        if obs is None:
            return None
        return abs(obs - self.expected_accuracy)


@dataclass
class ConfidenceCalibrationCurve:
    """
    Reliability diagram data for confidence calibration.
    
    Tracks the relationship between predicted confidence
    and observed accuracy across probability bins.
    """
    bins: List[ConfidenceBin] = field(default_factory=list)
    
    # Overall metrics
    expected_calibration_error: Optional[float] = None
    maximum_calibration_error: Optional[float] = None
    brier_score: Optional[float] = None
    
    def compute_metrics(self) -> None:
        """Compute calibration metrics from bins."""
        if not self.bins:
            return
        
        # Expected Calibration Error (ECE)
        total_samples = sum(b.total_samples for b in self.bins)
        if total_samples == 0:
            return
        
        ece = sum(
            (b.total_samples / total_samples) * (b.calibration_error() or 0)
            for b in self.bins
        )
        self.expected_calibration_error = ece
        
        # Maximum Calibration Error (MCE)
        self.maximum_calibration_error = max(
            (b.calibration_error() or 0) for b in self.bins
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bins": [
                {
                    "range": [b.bin_lower, b.bin_upper],
                    "center": b.bin_center,
                    "total_samples": b.total_samples,
                    "positive_samples": b.positive_samples,
                    "observed_accuracy": b.observed_accuracy,
                    "expected_accuracy": b.expected_accuracy,
                    "calibration_error": b.calibration_error()
                }
                for b in self.bins
            ],
            "metrics": {
                "expected_calibration_error": self.expected_calibration_error,
                "maximum_calibration_error": self.maximum_calibration_error,
                "brier_score": self.brier_score
            }
        }


@dataclass
class DomainApplicabilityDescriptor:
    """
    Descriptor for prediction domain applicability.
    
    Characterizes how similar a query compound is to the training
    distribution and whether the prediction is trustworthy.
    """
    
    # Distance metrics
    euclidean_distance_to_train: Optional[float] = None
    mahalanobis_distance: Optional[float] = None
    cosine_similarity_to_train: Optional[float] = None
    
    # Nearest neighbor information
    nearest_neighbor_distance: Optional[float] = None
    nearest_neighbor_identity: Optional[str] = None
    k_nearest_neighbor_distance: Optional[float] = None
    
    # Model-specific metrics
    ensemble_variance: Optional[float] = None  # For ensemble models
    dropout_variance: Optional[float] = None  # For MC-dropout
    gradient_norm: Optional[float] = None  # Gradient-based uncertainty
    
    # Domain classification
    domain_classification: str = "in_domain"  # in_domain, borderline, out_of_domain
    domain_confidence: Optional[float] = None  # Confidence in domain classification
    
    # Applicability domain method
    ad_method: str = "ensemble"  # kNN, leverage, ensemble, etc.
    ad_threshold: Optional[float] = None
    
    def is_in_domain(self) -> bool:
        """Check if prediction is within applicability domain."""
        return self.domain_classification == "in_domain"


@dataclass
class AIConfidenceCalibrationMetadata:
    """
    INNOVATION 2: AI Confidence Calibration Metadata
    
    Comprehensive metadata for calibrated confidence estimates in
    AI-powered virtual screening. Ensures that prediction uncertainties
    are meaningful and well-calibrated.
    
    Captures:
    - Calibration model version and parameters
    - Reliability curves (confidence vs. accuracy)
    - Domain applicability estimates
    - Uncertainty decomposition
    
    Purpose: Enable users to make informed decisions based on AI
    predictions by providing calibrated confidence intervals and
    domain applicability warnings.
    """
    
    # Model identification
    model_id: str
    model_version: str
    calibration_model_version: str
    
    # Calibration curves
    reliability_curve: ConfidenceCalibrationCurve = field(
        default_factory=lambda: ConfidenceCalibrationCurve()
    )
    
    # Domain applicability for this specific prediction
    domain_applicability: Optional[DomainApplicabilityDescriptor] = None
    
    # Uncertainty decomposition
    aleatoric_uncertainty: Optional[float] = None  # Data noise
    epistemic_uncertainty: Optional[float] = None  # Model uncertainty
    total_uncertainty: Optional[float] = None
    
    # Calibration method
    calibration_method: str = "isotonic"  # isotonic, platt_scaling, temperature_scaling, beta
    calibration_dataset_size: Optional[int] = None
    calibration_date: Optional[datetime] = None
    
    # Prediction-specific
    raw_prediction: Optional[float] = None
    calibrated_prediction: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_level: float = 0.95  # 95% confidence interval
    
    # Warnings
    calibration_warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def create_calibrated_prediction(
        cls,
        model_id: str,
        model_version: str,
        raw_prediction: float,
        calibration_curve: ConfidenceCalibrationCurve,
        domain_applicability: Optional[DomainApplicabilityDescriptor] = None
    ) -> "AIConfidenceCalibrationMetadata":
        """
        Create calibrated prediction with uncertainty estimates.
        
        This factory method applies calibration to a raw prediction
        and generates appropriate confidence intervals.
        """
        # Apply calibration based on reliability curve
        calibrated = cls._apply_calibration(raw_prediction, calibration_curve)
        
        # Compute confidence interval
        confidence_interval = cls._compute_confidence_interval(
            calibrated, calibration_curve
        )
        
        # Generate warnings
        warnings = []
        if domain_applicability and not domain_applicability.is_in_domain():
            warnings.append("Prediction outside applicability domain")
        
        if calibration_curve.expected_calibration_error:
            if calibration_curve.expected_calibration_error > 0.1:
                warnings.append("High calibration error detected")
        
        return cls(
            model_id=model_id,
            model_version=model_version,
            calibration_model_version="1.0.0",
            reliability_curve=calibration_curve,
            domain_applicability=domain_applicability,
            raw_prediction=raw_prediction,
            calibrated_prediction=calibrated,
            confidence_interval=confidence_interval,
            calibration_warnings=warnings,
            calibration_date=datetime.utcnow()
        )
    
    @staticmethod
    def _apply_calibration(
        raw_prediction: float,
        calibration_curve: ConfidenceCalibrationCurve
    ) -> float:
        """Apply calibration transformation to raw prediction."""
        # Find corresponding bin
        for bin in calibration_curve.bins:
            if bin.bin_lower <= raw_prediction < bin.bin_upper:
                # Adjust prediction based on observed accuracy
                if bin.observed_accuracy is not None:
                    # Blend raw prediction with observed accuracy
                    n = bin.total_samples
                    weight = min(n / 100, 0.5)  # Max 50% weight to observations
                    return (1 - weight) * raw_prediction + weight * bin.observed_accuracy
        
        return raw_prediction
    
    @staticmethod
    def _compute_confidence_interval(
        calibrated_prediction: float,
        calibration_curve: ConfidenceCalibrationCurve,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval based on calibration."""
        # Find corresponding bin
        for bin in calibration_curve.bins:
            if bin.bin_lower <= calibrated_prediction < bin.bin_upper:
                calibration_error = bin.calibration_error() or 0.1
                
                # Approximate confidence interval
                z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
                margin = z_score * calibration_error
                
                lower = max(0.0, calibrated_prediction - margin)
                upper = min(1.0, calibrated_prediction + margin)
                
                return (lower, upper)
        
        # Default wide interval if bin not found
        return (max(0.0, calibrated_prediction - 0.2), min(1.0, calibrated_prediction + 0.2))
    
    def get_reliability_assessment(self) -> Dict[str, Any]:
        """Generate human-readable reliability assessment."""
        assessment = {
            "prediction_reliability": "unknown",
            "confidence": self.calibrated_prediction,
            "confidence_interval": self.confidence_interval,
            "domain": self.domain_applicability.domain_classification if self.domain_applicability else "unknown",
            "warnings": self.calibration_warnings
        }
        
        # Determine reliability
        if self.domain_applicability and not self.domain_applicability.is_in_domain():
            assessment["prediction_reliability"] = "low"
        elif self.reliability_curve.expected_calibration_error:
            if self.reliability_curve.expected_calibration_error < 0.05:
                assessment["prediction_reliability"] = "high"
            elif self.reliability_curve.expected_calibration_error < 0.1:
                assessment["prediction_reliability"] = "medium"
            else:
                assessment["prediction_reliability"] = "low"
        else:
            assessment["prediction_reliability"] = "medium"
        
        return assessment
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "calibration_model_version": self.calibration_model_version,
            "calibration_method": self.calibration_method,
            "calibration_date": self.calibration_date.isoformat() if self.calibration_date else None,
            "reliability_curve": self.reliability_curve.to_dict(),
            "domain_applicability": {
                "domain_classification": self.domain_applicability.domain_classification if self.domain_applicability else None,
                "euclidean_distance": self.domain_applicability.euclidean_distance_to_train if self.domain_applicability else None,
                "ensemble_variance": self.domain_applicability.ensemble_variance if self.domain_applicability else None
            },
            "uncertainty": {
                "aleatoric": self.aleatoric_uncertainty,
                "epistemic": self.epistemic_uncertainty,
                "total": self.total_uncertainty
            },
            "prediction": {
                "raw": self.raw_prediction,
                "calibrated": self.calibrated_prediction,
                "confidence_interval": self.confidence_interval,
                "confidence_level": self.confidence_level
            },
            "warnings": self.calibration_warnings
        }


class CalibrationDatasetBuilder:
    """
    Builder for creating calibration datasets.
    
    Collects predictions and ground truth to build reliability curves
    and calibrate future predictions.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.predictions: List[Tuple[float, bool]] = []  # (prediction, ground_truth)
    
    def add_prediction(
        self,
        predicted_probability: float,
        actual_outcome: bool
    ) -> None:
        """Add a prediction with its ground truth."""
        self.predictions.append((predicted_probability, actual_outcome))
    
    def build_calibration_curve(self) -> ConfidenceCalibrationCurve:
        """Build calibration curve from collected predictions."""
        # Create bins
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bins = [
            ConfidenceBin(bin_edges[i], bin_edges[i + 1])
            for i in range(self.n_bins)
        ]
        
        # Populate bins
        for pred, actual in self.predictions:
            for bin in bins:
                if bin.bin_lower <= pred < bin.bin_upper:
                    bin.total_samples += 1
                    if actual:
                        bin.positive_samples += 1
                    break
        
        # Create curve
        curve = ConfidenceCalibrationCurve(bins=bins)
        curve.compute_metrics()
        
        return curve
    
    def compute_brier_score(self) -> float:
        """Compute Brier score for all predictions."""
        if not self.predictions:
            return 0.0
        
        return np.mean([
            (pred - float(actual)) ** 2
            for pred, actual in self.predictions
        ])


# Convenience functions

def generate_docking_reproducibility_hash(
    experiment: DockingExperimentMetadata,
    **kwargs
) -> DockingReproducibilityHash:
    """Convenience function to generate docking reproducibility hash."""
    return DockingReproducibilityHash.from_experiment(experiment, **kwargs)


def create_calibration_metadata(
    model_id: str,
    model_version: str,
    raw_prediction: float,
    calibration_predictions: List[Tuple[float, bool]],
    n_bins: int = 10
) -> AIConfidenceCalibrationMetadata:
    """
    Convenience function to create calibration metadata.
    
    Args:
        model_id: Model identifier
        model_version: Model version
        raw_prediction: Raw prediction to calibrate
        calibration_predictions: List of (prediction, ground_truth) for calibration
        n_bins: Number of bins for reliability curve
    """
    # Build calibration curve
    builder = CalibrationDatasetBuilder(n_bins=n_bins)
    for pred, actual in calibration_predictions:
        builder.add_prediction(pred, actual)
    
    curve = builder.build_calibration_curve()
    curve.brier_score = builder.compute_brier_score()
    
    return AIConfidenceCalibrationMetadata.create_calibrated_prediction(
        model_id=model_id,
        model_version=model_version,
        raw_prediction=raw_prediction,
        calibration_curve=curve
    )
