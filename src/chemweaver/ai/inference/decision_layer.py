"""
AISUAM - Screening Decision Layer
==================================

Uncertainty-calibrated screening decisions integrating predictions,
uncertainty estimates, and domain applicability into actionable
screening recommendations.

Author: AISUAM Development Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np


class ScreeningDecision(Enum):
    """Possible screening decisions."""
    PASS = "pass"           # Pass to next stage
    FAIL = "fail"           # Eliminate from pipeline
    REVIEW = "review"       # Flag for expert review
    REPLICATE = "replicate" # Request additional data
    SKIP = "skip"          # Skip to later stage (fast track)


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""
    VERY_HIGH = 0.95
    HIGH = 0.85
    MEDIUM = 0.70
    LOW = 0.50
    VERY_LOW = 0.30


@dataclass
class ScreeningRecommendation:
    """
    Complete screening recommendation for a compound.
    
    Combines prediction, uncertainty, applicability, and decision.
    """
    
    # Compound identification
    compound_id: UUID
    
    # Predictions
    predicted_docking_score: Optional[float] = None
    predicted_activity_probability: Optional[float] = None
    predicted_rank: Optional[int] = None
    
    # Uncertainty
    score_uncertainty: float = 0.5
    confidence_interval_95: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    
    # Applicability
    in_domain: bool = True
    domain_reliability: float = 0.5
    nearest_training_distance: Optional[float] = None
    
    # Decision
    decision: ScreeningDecision = ScreeningDecision.REVIEW
    decision_confidence: float = 0.5
    decision_rationale: str = ""
    
    # Screening metadata
    stage_id: Optional[UUID] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Alternative scenarios
    alternative_decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compound_id": str(self.compound_id),
            "predictions": {
                "docking_score": self.predicted_docking_score,
                "activity_probability": self.predicted_activity_probability,
                "rank": self.predicted_rank
            },
            "uncertainty": {
                "std": self.score_uncertainty,
                "ci_95": self.confidence_interval_95
            },
            "applicability": {
                "in_domain": self.in_domain,
                "reliability": self.domain_reliability,
                "distance": self.nearest_training_distance
            },
            "decision": {
                "action": self.decision.value,
                "confidence": self.decision_confidence,
                "rationale": self.decision_rationale
            }
        }
    
    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if recommendation is reliable enough for automatic decision."""
        return (
            self.in_domain and
            self.decision_confidence >= threshold and
            self.domain_reliability >= threshold
        )


class AdaptiveConfidenceThresholding:
    """
    Adaptive confidence thresholds based on model uncertainty and
    screening stage requirements.
    
    Dynamically adjusts thresholds based on:
    - Model uncertainty
    - Target difficulty
    - Resource constraints
    - Historical performance
    """
    
    def __init__(
        self,
        base_threshold: float = 0.7,
        target_difficulty: str = "moderate"
    ):
        self.base_threshold = base_threshold
        self.target_difficulty = target_difficulty
        self.threshold_history: List[float] = []
        
        # Adjust base threshold by difficulty
        self.difficulty_adjustments = {
            "easy": -0.1,
            "moderate": 0.0,
            "hard": 0.1,
            "very_hard": 0.2
        }
    
    def compute_threshold(
        self,
        model_uncertainty: float,
        stage_requirements: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute adaptive threshold for current conditions.
        
        Higher uncertainty → Lower threshold (more permissive)
        Harder target → Higher threshold (more selective)
        """
        # Start with base threshold
        threshold = self.base_threshold
        
        # Adjust for target difficulty
        difficulty_adj = self.difficulty_adjustments.get(self.target_difficulty, 0.0)
        threshold += difficulty_adj
        
        # Adjust for model uncertainty
        # High uncertainty → need higher confidence to compensate
        uncertainty_adj = model_uncertainty * 0.2
        threshold += uncertainty_adj
        
        # Adjust for stage requirements
        if stage_requirements:
            precision_weight = stage_requirements.get("precision_weight", 0.5)
            recall_weight = stage_requirements.get("recall_weight", 0.5)
            
            # Balance between precision and recall
            if precision_weight > recall_weight:
                threshold += 0.05  # More selective
            else:
                threshold -= 0.05  # More permissive
        
        # Clamp to valid range
        threshold = max(0.3, min(0.95, threshold))
        
        self.threshold_history.append(threshold)
        return threshold
    
    def get_recommended_action(
        self,
        prediction: float,
        uncertainty: float,
        applicability: float,
        in_domain: bool
    ) -> Tuple[ScreeningDecision, float, str]:
        """
        Get recommended screening action.
        
        Returns decision, confidence, and rationale.
        """
        # Compute adaptive threshold
        threshold = self.compute_threshold(uncertainty)
        
        # Decision logic
        if not in_domain:
            # Out of domain compounds
            if uncertainty > 0.5:
                return (
                    ScreeningDecision.REVIEW,
                    0.6,
                    "Out of domain with high uncertainty - requires expert review"
                )
            else:
                return (
                    ScreeningDecision.FAIL,
                    0.7,
                    "Out of domain - cannot reliably assess"
                )
        
        # In-domain compounds
        confidence = (1 - uncertainty) * applicability
        
        if prediction >= threshold and confidence >= 0.8:
            return (
                ScreeningDecision.PASS,
                confidence,
                f"High confidence prediction ({prediction:.2f}) above threshold ({threshold:.2f})"
            )
        elif prediction >= threshold and confidence >= 0.6:
            return (
                ScreeningDecision.PASS,
                confidence,
                f"Moderate confidence prediction ({prediction:.2f}) above threshold ({threshold:.2f})"
            )
        elif prediction < threshold and confidence >= 0.7:
            return (
                ScreeningDecision.FAIL,
                confidence,
                f"High confidence prediction ({prediction:.2f}) below threshold ({threshold:.2f})"
            )
        elif uncertainty > 0.4:
            return (
                ScreeningDecision.REPLICATE,
                0.5,
                "High uncertainty - recommend additional data collection"
            )
        else:
            return (
                ScreeningDecision.REVIEW,
                0.5,
                "Uncertain prediction - requires manual review"
            )


class UncertaintyCalibratedDecisionLayer:
    """
    Main decision layer integrating all AISUAM outputs.
    
    Combines:
    - Docking surrogate predictions
    - Activity probability predictions  
    - Uncertainty estimates (aleatoric + epistemic)
    - Domain applicability
    - Adaptive thresholds
    
    Outputs complete screening recommendations.
    """
    
    def __init__(
        self,
        base_threshold: float = 0.7,
        target_difficulty: str = "moderate"
    ):
        self.layer_id = uuid4()
        self.thresholding = AdaptiveConfidenceThresholding(
            base_threshold, target_difficulty
        )
        self.decision_history: List[ScreeningRecommendation] = []
    
    def make_decision(
        self,
        compound_id: UUID,
        docking_prediction: Optional[float] = None,
        activity_prediction: Optional[float] = None,
        uncertainty: Optional[float] = None,
        in_domain: bool = True,
        domain_reliability: float = 0.5,
        stage_id: Optional[UUID] = None
    ) -> ScreeningRecommendation:
        """
        Make comprehensive screening decision.
        
        Integrates all available information into actionable recommendation.
        """
        # Default values
        if uncertainty is None:
            uncertainty = 0.5
        
        # Combine predictions if both available
        if docking_prediction is not None and activity_prediction is not None:
            # Weighted combination (docking is primary)
            combined_score = 0.7 * docking_prediction + 0.3 * activity_prediction
        elif docking_prediction is not None:
            combined_score = docking_prediction
        elif activity_prediction is not None:
            combined_score = activity_prediction
        else:
            combined_score = 0.5
        
        # Compute confidence interval
        ci_lower = combined_score - 1.96 * uncertainty
        ci_upper = combined_score + 1.96 * uncertainty
        
        # Get decision from thresholding module
        decision, decision_confidence, rationale = \
            self.thresholding.get_recommended_action(
                prediction=combined_score,
                uncertainty=uncertainty,
                applicability=domain_reliability,
                in_domain=in_domain
            )
        
        # Create recommendation
        recommendation = ScreeningRecommendation(
            compound_id=compound_id,
            predicted_docking_score=docking_prediction,
            predicted_activity_probability=activity_prediction,
            score_uncertainty=uncertainty,
            confidence_interval_95=(ci_lower, ci_upper),
            in_domain=in_domain,
            domain_reliability=domain_reliability,
            decision=decision,
            decision_confidence=decision_confidence,
            decision_rationale=rationale,
            stage_id=stage_id
        )
        
        self.decision_history.append(recommendation)
        return recommendation
    
    def batch_decisions(
        self,
        compound_ids: List[UUID],
        predictions: Dict[str, List[float]],
        uncertainties: List[float],
        in_domain: List[bool],
        domain_reliability: List[float],
        stage_id: Optional[UUID] = None
    ) -> List[ScreeningRecommendation]:
        """Make decisions for a batch of compounds."""
        recommendations = []
        
        for i, cid in enumerate(compound_ids):
            rec = self.make_decision(
                compound_id=cid,
                docking_prediction=predictions.get("docking", [None])[i],
                activity_prediction=predictions.get("activity", [None])[i],
                uncertainty=uncertainties[i],
                in_domain=in_domain[i],
                domain_reliability=domain_reliability[i],
                stage_id=stage_id
            )
            recommendations.append(rec)
        
        return recommendations
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics on decisions made."""
        if not self.decision_history:
            return {}
        
        decisions = [r.decision for r in self.decision_history]
        
        return {
            "total_decisions": len(decisions),
            "pass_count": sum(1 for d in decisions if d == ScreeningDecision.PASS),
            "fail_count": sum(1 for d in decisions if d == ScreeningDecision.FAIL),
            "review_count": sum(1 for d in decisions if d == ScreeningDecision.REVIEW),
            "replicate_count": sum(1 for d in decisions if d == ScreeningDecision.REPLICATE),
            "avg_confidence": np.mean([r.decision_confidence for r in self.decision_history]),
            "avg_uncertainty": np.mean([r.score_uncertainty for r in self.decision_history])
        }
    
    def generate_decision_report(self) -> str:
        """Generate human-readable decision report."""
        stats = self.get_decision_statistics()
        
        lines = [
            "=" * 70,
            "UNCERTAINTY-CALIBRATED SCREENING DECISION REPORT",
            "=" * 70,
            "",
            f"Layer ID: {self.layer_id}",
            f"Total Decisions: {stats.get('total_decisions', 0)}",
            "",
            "DECISION BREAKDOWN",
            "-" * 40,
            f"  PASS:      {stats.get('pass_count', 0)}",
            f"  FAIL:      {stats.get('fail_count', 0)}",
            f"  REVIEW:    {stats.get('review_count', 0)}",
            f"  REPLICATE: {stats.get('replicate_count', 0)}",
            "",
            "UNCERTAINTY METRICS",
            "-" * 40,
            f"  Average Confidence:  {stats.get('avg_confidence', 0):.2%}",
            f"  Average Uncertainty: {stats.get('avg_uncertainty', 0):.2%}",
            "",
            "RECENT DECISIONS",
            "-" * 40
        ]
        
        # Show last 5 decisions
        for rec in self.decision_history[-5:]:
            lines.append(f"\n  {rec.compound_id.hex[:8]}...")
            lines.append(f"    Action: {rec.decision.value}")
            lines.append(f"    Score: {rec.predicted_docking_score or rec.predicted_activity_probability:.2f} ± {rec.score_uncertainty:.2f}")
            lines.append(f"    In Domain: {rec.in_domain}")
            lines.append(f"    Confidence: {rec.decision_confidence:.2%}")
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)


# Export
__all__ = [
    'ScreeningDecision',
    'ConfidenceLevel',
    'ScreeningRecommendation',
    'AdaptiveConfidenceThresholding',
    'UncertaintyCalibratedDecisionLayer'
]
