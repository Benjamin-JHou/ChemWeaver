"""
AISUAM Innovations Module
==========================

Mandatory Innovation Elements:
1. Joint Docking-Activity Multi-Task Learning
2. Physics-Regularized Loss Functions

Additional Innovations:
3. Screening Uncertainty Propagation Modeling
4. Adaptive Screening Confidence Thresholding

Author: AISUAM Development Team
Version: 1.0.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4


# ============================================================================
# INNOVATION 1: Joint Docking-Activity Multi-Task Learning
# ============================================================================

class JointDockingActivityLearning(nn.Module):
    """
    Joint learning of docking scores and activity predictions.
    
    Motivation:
    - Docking scores reflect physics-based binding
    - Activity predictions reflect experimental observations
    - Joint learning leverages complementary information
    - Improves both tasks through shared representations
    
    Architecture:
    - Shared encoder (GNN + fusion)
    - Task-specific heads with cross-task attention
    - Auxiliary task connections
    """
    
    def __init__(
        self,
        shared_encoder: nn.Module,
        docking_head: nn.Module,
        activity_head: nn.Module,
        shared_dim: int = 128,
        use_cross_attention: bool = True
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.docking_head = docking_head
        self.activity_head = activity_head
        self.use_cross_attention = use_cross_attention
        
        # Cross-task attention
        if use_cross_attention:
            self.docking_to_activity = nn.Linear(shared_dim, shared_dim)
            self.activity_to_docking = nn.Linear(shared_dim, shared_dim)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=shared_dim,
                num_heads=4,
                batch_first=True
            )
        
        # Task relationship modeling
        self.task_correlation = nn.Parameter(torch.eye(2))  # 2x2 correlation matrix
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        fingerprint: torch.Tensor,
        descriptor: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with joint prediction.
        
        Returns:
            docking_score: [batch, 1]
            activity_prob: [batch, 1]
            uncertainty: [batch, 2] (optional)
        """
        # Shared encoding
        shared_repr = self.shared_encoder(
            node_features, edge_index, fingerprint, descriptor, edge_features
        )
        
        if self.use_cross_attention:
            # Cross-task information flow
            docking_query = self.docking_to_activity(shared_repr)
            activity_query = self.activity_to_docking(shared_repr)
            
            # Stack for attention
            queries = torch.stack([docking_query, activity_query], dim=1)
            
            # Cross-attention
            attn_output, attn_weights = self.cross_attention(queries, queries, queries)
            
            # Split back
            docking_enhanced = attn_output[:, 0] + shared_repr
            activity_enhanced = attn_output[:, 1] + shared_repr
        else:
            docking_enhanced = shared_repr
            activity_enhanced = shared_repr
        
        # Task-specific predictions
        docking_score = self.docking_head(docking_enhanced)
        activity_prob = torch.sigmoid(self.activity_head(activity_enhanced))
        
        # Concatenate outputs
        output = torch.cat([docking_score, activity_prob], dim=-1)
        
        if return_attention and self.use_cross_attention:
            return output, attn_weights, self.task_correlation
        
        return output
    
    def compute_joint_loss(
        self,
        predictions: torch.Tensor,
        docking_targets: torch.Tensor,
        activity_targets: torch.Tensor,
        task_weights: List[float] = [1.0, 1.0],
        consistency_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute joint loss with task consistency.
        
        Loss = w1 * L_docking + w2 * L_activity + w3 * L_consistency
        """
        # Task-specific losses
        docking_loss = F.mse_loss(predictions[:, 0], docking_targets)
        activity_loss = F.binary_cross_entropy(predictions[:, 1], activity_targets)
        
        # Consistency loss: activity should correlate with good docking scores
        # Normalize docking scores to [0, 1] for comparison
        docking_normalized = torch.sigmoid(-predictions[:, 0] / 3.0)  # -9 kcal/mol → 0.95
        consistency_loss = F.mse_loss(docking_normalized, activity_targets)
        
        # Weighted sum
        total_loss = (
            task_weights[0] * docking_loss +
            task_weights[1] * activity_loss +
            consistency_weight * consistency_loss
        )
        
        return total_loss
    
    def predict_with_consistency_check(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        fingerprint: torch.Tensor,
        descriptor: torch.Tensor,
        consistency_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Predict with consistency checking between tasks.
        
        Returns predictions only if docking and activity agree.
        """
        output = self.forward(
            node_features, edge_index, fingerprint, descriptor
        )
        
        docking_score = output[:, 0].item()
        activity_prob = output[:, 1].item()
        
        # Check consistency
        # Good docking (< -7 kcal/mol) should correlate with high activity (> 0.5)
        docking_good = docking_score < -7.0
        activity_good = activity_prob > 0.5
        
        consistent = (docking_good and activity_good) or (not docking_good and not activity_good)
        
        return {
            "docking_score": docking_score,
            "activity_probability": activity_prob,
            "consistent": consistent,
            "confidence": 0.9 if consistent else 0.5
        }


# ============================================================================
# INNOVATION 2: Physics-Regularized Loss Functions
# ============================================================================

class PhysicsRegularizedLoss:
    """
    Loss functions incorporating physical constraints from docking.
    
    Motivation:
    - Pure ML losses ignore physical binding principles
    - Physics constraints improve generalization
    - Enforce physically plausible predictions
    
    Constraints:
    - Score should reflect binding free energy
    - Similar poses → Similar scores
    - Consistent with force field energies
    """
    
    def __init__(
        self,
        physics_weight: float = 0.1,
        temperature: float = 298.15,  # Kelvin
        use_force_field_consistency: bool = True
    ):
        self.physics_weight = physics_weight
        self.temperature = temperature
        self.kT = 0.001987 * temperature  # kcal/mol
        self.use_force_field_consistency = use_force_field_consistency
    
    def binding_free_energy_loss(
        self,
        predicted_scores: torch.Tensor,
        experimental_affinities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Loss ensuring predicted scores relate to binding free energy.
        
        ΔG ≈ RT ln(Kd) ≈ docking_score (with appropriate scaling)
        """
        # Convert scores to approximate ΔG
        # Typical scaling: docking_score (kcal/mol) ≈ ΔG
        delta_g = predicted_scores
        
        if experimental_affinities is not None:
            # Supervised loss
            loss = F.mse_loss(delta_g, experimental_affinities)
        else:
            # Unsupervised: encourage physically reasonable range
            # Binding free energies typically -5 to -15 kcal/mol
            reasonable_range = torch.relu(torch.abs(delta_g + 10) - 5)
            loss = reasonable_range.mean()
        
        return loss
    
    def pose_consistency_loss(
        self,
        scores: torch.Tensor,
        poses: List[Dict[str, Any]],
        rmsd_threshold: float = 2.0
    ) -> torch.Tensor:
        """
        Loss enforcing similar scores for similar poses.
        
        RMSD < threshold → |score1 - score2| < epsilon
        """
        if len(poses) < 2:
            return torch.tensor(0.0)
        
        consistency_loss = 0.0
        count = 0
        
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                rmsd = poses[i].get("rmsd_to", {}).get(j, float('inf'))
                
                if rmsd < rmsd_threshold:
                    # Similar poses should have similar scores
                    score_diff = torch.abs(scores[i] - scores[j])
                    consistency_loss += torch.relu(score_diff - 1.0)  # Allow 1 kcal/mol difference
                    count += 1
        
        return consistency_loss / max(count, 1)
    
    def interaction_fingerprint_loss(
        self,
        predicted_scores: torch.Tensor,
        interaction_fingerprints: torch.Tensor,
        target_interactions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Loss based on interaction fingerprint similarity.
        
        Similar interaction patterns → Similar binding affinities
        """
        # Compute pairwise interaction similarity
        if len(predicted_scores) < 2:
            return torch.tensor(0.0)
        
        # Cosine similarity of interaction fingerprints
        normalized_fp = F.normalize(interaction_fingerprints, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_fp, normalized_fp.t())
        
        # Score differences
        score_diff_matrix = torch.abs(
            predicted_scores.unsqueeze(1) - predicted_scores.unsqueeze(0)
        )
        
        # Similar interactions should have similar scores
        # High similarity (close to 1) → Low score difference
        consistency = similarity_matrix * (1.0 - torch.sigmoid(score_diff_matrix - 2.0))
        
        # Maximize consistency
        loss = 1.0 - consistency.mean()
        
        return loss
    
    def force_field_consistency_loss(
        self,
        predicted_scores: torch.Tensor,
        force_field_energies: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Loss enforcing consistency with force field calculations.
        
        Predicted scores should correlate with FF energies.
        """
        if weights is None:
            weights = torch.ones_like(predicted_scores)
        
        # Normalize both to have similar scale
        pred_norm = (predicted_scores - predicted_scores.mean()) / (predicted_scores.std() + 1e-8)
        ff_norm = (force_field_energies - force_field_energies.mean()) / (force_field_energies.std() + 1e-8)
        
        # Correlation loss (negative correlation to maximize)
        correlation = (pred_norm * ff_norm * weights).sum() / weights.sum()
        
        return 1.0 - correlation  # Minimize negative correlation
    
    def compute_total_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        poses: Optional[List[Dict]] = None,
        interaction_fps: Optional[torch.Tensor] = None,
        ff_energies: Optional[torch.Tensor] = None,
        base_loss_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total physics-regularized loss.
        
        Returns dict with individual loss components.
        """
        losses = {}
        
        # Base prediction loss
        losses["base"] = F.mse_loss(predictions, targets)
        
        # Physics constraints
        if self.use_force_field_consistency:
            losses["binding_energy"] = self.binding_free_energy_loss(predictions, targets)
            
            if poses is not None:
                losses["pose_consistency"] = self.pose_consistency_loss(predictions, poses)
            
            if interaction_fps is not None:
                losses["interaction_fp"] = self.interaction_fingerprint_loss(
                    predictions, interaction_fps
                )
            
            if ff_energies is not None:
                losses["force_field"] = self.force_field_consistency_loss(
                    predictions, ff_energies
                )
        
        # Total weighted loss
        total = base_loss_weight * losses["base"]
        for key, value in losses.items():
            if key != "base":
                total += self.physics_weight * value
        
        losses["total"] = total
        
        return losses


# ============================================================================
# INNOVATION 3: Screening Uncertainty Propagation Modeling
# ============================================================================

@dataclass
class ScreeningStageUncertainty:
    """Uncertainty tracking for a screening stage."""
    stage_id: UUID
    stage_name: str
    input_uncertainty: float
    model_uncertainty: float
    output_uncertainty: float
    propagated_from_previous: float = 0.0
    
    def compute_total(self) -> float:
        """Compute total uncertainty including propagation."""
        return np.sqrt(
            self.input_uncertainty**2 +
            self.model_uncertainty**2 +
            self.propagated_from_previous**2
        )


class ScreeningUncertaintyPropagation:
    """
    Model uncertainty propagation across screening stages.
    
    Each stage adds uncertainty, and filtering operations
    may reduce uncertainty by selecting more reliable predictions.
    """
    
    def __init__(self, pipeline_id: UUID):
        self.pipeline_id = pipeline_id
        self.stage_uncertainties: Dict[UUID, ScreeningStageUncertainty] = {}
        self.propagation_graph: List[Tuple[UUID, UUID]] = []  # (from, to)
    
    def register_stage(
        self,
        stage_id: UUID,
        stage_name: str,
        base_model_uncertainty: float
    ) -> ScreeningStageUncertainty:
        """Register a screening stage."""
        unc = ScreeningStageUncertainty(
            stage_id=stage_id,
            stage_name=stage_name,
            input_uncertainty=0.0,  # Will be set by propagation
            model_uncertainty=base_model_uncertainty,
            output_uncertainty=base_model_uncertainty
        )
        self.stage_uncertainties[stage_id] = unc
        return unc
    
    def propagate(
        self,
        from_stage_id: UUID,
        to_stage_id: UUID,
        propagation_factor: float = 1.0,
        filtering_enhancement: float = 0.0
    ) -> ScreeningStageUncertainty:
        """
        Propagate uncertainty from one stage to the next.
        
        Args:
            propagation_factor: How much uncertainty scales (1.0 = same)
            filtering_enhancement: Uncertainty reduction from filtering (0-1)
        """
        if from_stage_id not in self.stage_uncertainties:
            raise ValueError(f"Source stage {from_stage_id} not registered")
        if to_stage_id not in self.stage_uncertainties:
            raise ValueError(f"Target stage {to_stage_id} not registered")
        
        source = self.stage_uncertainties[from_stage_id]
        target = self.stage_uncertainties[to_stage_id]
        
        # Propagate output uncertainty from source
        propagated = source.output_uncertainty * propagation_factor
        
        # Apply filtering enhancement if applicable
        if filtering_enhancement > 0:
            propagated *= (1 - filtering_enhancement)
        
        # Set as input uncertainty for target
        target.input_uncertainty = propagated
        target.propagated_from_previous = propagated
        
        # Compute new output uncertainty
        target.output_uncertainty = target.compute_total()
        
        # Record propagation
        self.propagation_graph.append((from_stage_id, to_stage_id))
        
        return target
    
    def get_pipeline_uncertainty(self) -> float:
        """Get total accumulated uncertainty across pipeline."""
        if not self.stage_uncertainties:
            return 0.0
        
        # Sum variances, take sqrt
        total_variance = sum(
            s.output_uncertainty**2 for s in self.stage_uncertainties.values()
        )
        return np.sqrt(total_variance)
    
    def get_compound_uncertainty(
        self,
        compound_id: UUID,
        stage_id: UUID
    ) -> Dict[str, float]:
        """Get uncertainty breakdown for a specific compound."""
        if stage_id not in self.stage_uncertainties:
            return {"error": "Stage not found"}
        
        stage_unc = self.stage_uncertainties[stage_id]
        
        return {
            "compound_id": str(compound_id),
            "stage": stage_unc.stage_name,
            "input_uncertainty": stage_unc.input_uncertainty,
            "model_uncertainty": stage_unc.model_uncertainty,
            "propagated": stage_unc.propagated_from_previous,
            "total": stage_unc.output_uncertainty
        }


# ============================================================================
# INNOVATION 4: Adaptive Screening Confidence Thresholding
# ============================================================================

class AdaptiveConfidenceThresholdingInnovation:
    """
    Advanced adaptive thresholding with multiple strategies.
    
    Goes beyond simple threshold adjustment to include:
    - Cost-aware thresholding
    - Multi-objective optimization
    - Dynamic strategy switching
    """
    
    def __init__(self):
        self.strategy_id = uuid4()
        self.threshold_history: List[Dict[str, Any]] = []
    
    def cost_aware_threshold(
        self,
        base_threshold: float,
        remaining_budget: float,
        total_budget: float,
        cost_per_compound: float,
        expected_hits: int
    ) -> float:
        """
        Adjust threshold based on cost constraints.
        
        If budget is tight, lower threshold to select more candidates.
        """
        budget_ratio = remaining_budget / total_budget
        
        if budget_ratio < 0.2:
            # Low budget: be more permissive
            adjustment = -0.15
        elif budget_ratio < 0.5:
            # Medium budget: slight adjustment
            adjustment = -0.05
        else:
            # Good budget: strict threshold
            adjustment = 0.0
        
        new_threshold = base_threshold + adjustment
        
        self.threshold_history.append({
            "type": "cost_aware",
            "base": base_threshold,
            "adjusted": new_threshold,
            "budget_ratio": budget_ratio
        })
        
        return max(0.3, min(0.95, new_threshold))
    
    def multi_objective_threshold(
        self,
        precision_weight: float = 0.5,
        recall_weight: float = 0.5,
        enrichment_weight: float = 0.0,
        current_precision: Optional[float] = None,
        current_recall: Optional[float] = None
    ) -> float:
        """
        Optimize threshold for multiple objectives.
        
        Balances precision, recall, and enrichment.
        """
        # Higher precision → higher threshold
        # Higher recall → lower threshold
        
        base = 0.7
        
        if current_precision is not None and current_recall is not None:
            # Adjust based on current performance
            if precision_weight > recall_weight:
                # Prioritize precision
                if current_precision < 0.8:
                    adjustment = 0.1
                else:
                    adjustment = 0.0
            else:
                # Prioritize recall
                if current_recall < 0.5:
                    adjustment = -0.1
                else:
                    adjustment = 0.0
        else:
            # Use weights directly
            adjustment = (precision_weight - recall_weight) * 0.2
        
        new_threshold = base + adjustment
        
        self.threshold_history.append({
            "type": "multi_objective",
            "base": base,
            "adjusted": new_threshold,
            "weights": {
                "precision": precision_weight,
                "recall": recall_weight,
                "enrichment": enrichment_weight
            }
        })
        
        return max(0.3, min(0.95, new_threshold))
    
    def get_threshold_statistics(self) -> Dict[str, Any]:
        """Get statistics on threshold adaptations."""
        if not self.threshold_history:
            return {}
        
        adjustments = [
            h["adjusted"] - h["base"]
            for h in self.threshold_history
        ]
        
        return {
            "total_adjustments": len(self.threshold_history),
            "avg_adjustment": sum(adjustments) / len(adjustments),
            "max_increase": max(adjustments),
            "max_decrease": min(adjustments),
            "adjustment_types": list(set(h["type"] for h in self.threshold_history))
        }


# Export
__all__ = [
    'JointDockingActivityLearning',
    'PhysicsRegularizedLoss',
    'ScreeningUncertaintyPropagation',
    'AdaptiveConfidenceThresholdingInnovation'
]
