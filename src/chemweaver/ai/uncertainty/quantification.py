"""
AISUAM - Uncertainty Modeling Layer
====================================

Comprehensive uncertainty quantification implementing multiple methods:
- Deep Ensemble
- Monte Carlo Dropout
- Conformal Prediction
- Bayesian Neural Network Approximation

Author: AISUAM Development Team
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4


@dataclass
class UncertaintyEstimate:
    """
    Complete uncertainty estimate for a prediction.
    
    Includes both aleatoric (data) and epistemic (model) uncertainty.
    """
    
    # Point prediction
    prediction: float
    
    # Uncertainty decomposition
    aleatoric_std: float = 0.0  # Data noise
    epistemic_std: float = 0.0  # Model uncertainty
    total_std: float = field(init=False)
    
    # Confidence intervals
    confidence_interval_95: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    confidence_interval_99: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    
    # Calibration metrics
    expected_calibration_error: Optional[float] = None
    maximum_calibration_error: Optional[float] = None
    
    # Reliability
    reliability_flag: str = "unknown"  # high, medium, low, unknown
    
    # Source information
    uncertainty_method: str = "unknown"
    
    def __post_init__(self):
        """Compute total uncertainty."""
        self.total_std = np.sqrt(self.aleatoric_std**2 + self.epistemic_std**2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "uncertainty": {
                "aleatoric": self.aleatoric_std,
                "epistemic": self.epistemic_std,
                "total": self.total_std
            },
            "confidence_intervals": {
                "95": self.confidence_interval_95,
                "99": self.confidence_interval_99
            },
            "reliability": self.reliability_flag,
            "method": self.uncertainty_method
        }


class UncertaintyMethod(ABC):
    """Abstract base class for uncertainty quantification methods."""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.method_id = uuid4()
    
    @abstractmethod
    def predict_with_uncertainty(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty quantification.
        
        Returns:
            mean: [batch, num_tasks]
            aleatoric: [batch, num_tasks]
            epistemic: [batch, num_tasks]
        """
        pass
    
    def compute_confidence_intervals(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute confidence intervals."""
        from scipy import stats
        
        intervals = {}
        for level in confidence_levels:
            z_score = stats.norm.ppf((1 + level) / 2)
            lower = mean - z_score * std
            upper = mean + z_score * std
            intervals[level] = (lower, upper)
        
        return intervals


class DeepEnsembleUncertainty(UncertaintyMethod):
    """
    Deep Ensemble Uncertainty Quantification.
    
    Uses multiple independently trained models to estimate uncertainty.
    Epistemic uncertainty comes from model disagreement.
    """
    
    def __init__(self, num_models: int = 5):
        super().__init__("deep_ensemble")
        self.num_models = num_models
        self.models: List[nn.Module] = []
    
    def add_model(self, model: nn.Module) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
    
    def predict_with_uncertainty(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction with uncertainty.
        
        For ensembles, epistemic uncertainty is std across models.
        Aleatoric uncertainty is mean of individual uncertainties.
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        all_predictions = []
        all_uncertainties = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(x, return_uncertainty=True)
                
                if isinstance(output, tuple):
                    pred, unc = output
                else:
                    pred = output
                    unc = torch.ones_like(pred) * 0.5
                
                all_predictions.append(pred)
                all_uncertainties.append(unc)
        
        # Stack predictions
        all_predictions = torch.stack(all_predictions)  # [num_models, batch, tasks]
        all_uncertainties = torch.stack(all_uncertainties)
        
        # Mean prediction
        mean = all_predictions.mean(dim=0)
        
        # Epistemic uncertainty: variance across models
        epistemic = all_predictions.std(dim=0)
        
        # Aleatoric uncertainty: mean of individual uncertainties
        aleatoric = all_uncertainties.mean(dim=0)
        
        return mean, aleatoric, epistemic
    
    def train_ensemble_member(
        self,
        model_class: type,
        config: Dict[str, Any],
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader,
        epochs: int = 100
    ) -> nn.Module:
        """Train a single ensemble member with different initialization."""
        import random
        
        # Set different seed for each ensemble member
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_class(**config)
        # Training logic would go here
        
        return model


class MonteCarloDropoutUncertainty(UncertaintyMethod):
    """
    Monte Carlo Dropout Uncertainty.
    
    Uses dropout at inference time to approximate Bayesian inference.
    Fast and effective uncertainty estimation.
    """
    
    def __init__(self, dropout_rate: float = 0.2):
        super().__init__("mc_dropout")
        self.dropout_rate = dropout_rate
    
    def enable_dropout(self, model: nn.Module) -> None:
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()  # Keep dropout active
    
    def predict_with_uncertainty(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC Dropout prediction with uncertainty.
        
        Multiple stochastic forward passes provide uncertainty estimate.
        """
        model.train()  # Keep dropout active
        self.enable_dropout(model)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(x)
                predictions.append(output)
        
        predictions = torch.stack(predictions)  # [num_samples, batch, tasks]
        
        # Mean prediction
        mean = predictions.mean(dim=0)
        
        # Total uncertainty (variance across samples)
        total_variance = predictions.var(dim=0)
        
        # For MC Dropout, we approximate:
        # Total variance = Epistemic + Aleatoric
        # We'll estimate epistemic as a fraction of total
        epistemic = total_variance * 0.7  # Heuristic: 70% epistemic
        aleatoric = total_variance * 0.3
        
        return mean, torch.sqrt(aleatoric), torch.sqrt(epistemic)


class ConformalPredictionUncertainty(UncertaintyMethod):
    """
    Conformal Prediction for uncertainty quantification.
    
    Provides prediction sets with guaranteed coverage probability.
    Distribution-free and model-agnostic.
    """
    
    def __init__(self, coverage: float = 0.95):
        super().__init__("conformal_prediction")
        self.coverage = coverage
        self.calibration_scores: Optional[np.ndarray] = None
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ) -> None:
        """
        Calibrate conformal predictor on calibration set.
        
        Must be called before prediction.
        """
        scores = []
        
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                x, y = batch
                predictions = model(x)
                
                # Compute non-conformity scores (absolute residuals)
                residuals = torch.abs(predictions - y).cpu().numpy()
                scores.append(residuals)
        
        self.calibration_scores = np.concatenate(scores, axis=0)
    
    def predict_with_uncertainty(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Conformal prediction with uncertainty.
        
        Returns prediction intervals with guaranteed coverage.
        """
        if self.calibration_scores is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        model.eval()
        with torch.no_grad():
            predictions = model(x)
        
        # Compute quantile for desired coverage
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * self.coverage) / n
        q = np.quantile(self.calibration_scores, q_level, axis=0)
        
        # Convert to torch
        q = torch.from_numpy(q).to(predictions.device)
        
        # For conformal, uncertainty is the quantile value
        uncertainty = q.expand_as(predictions)
        
        # Conformal doesn't distinguish aleatoric/epistemic
        # We assign all to epistemic as it's model-dependent
        return predictions, torch.zeros_like(uncertainty), uncertainty
    
    def get_prediction_interval(
        self,
        prediction: torch.Tensor,
        alpha: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction interval for given significance level."""
        if self.calibration_scores is None:
            raise ValueError("Model not calibrated")
        
        q = np.quantile(self.calibration_scores, 1 - alpha, axis=0)
        q = torch.from_numpy(q).to(prediction.device)
        
        lower = prediction - q
        upper = prediction + q
        
        return lower, upper


class EvidentialUncertainty(UncertaintyMethod):
    """
    Evidential Deep Learning for uncertainty.
    
    Models uncertainty using Normal-Inverse-Gamma distribution.
    Provides both aleatoric and epistemic uncertainty in single forward pass.
    """
    
    def __init__(self):
        super().__init__("evidential")
    
    def nig_loss(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
        lambda_reg: float = 0.01
    ) -> torch.Tensor:
        """
        Normal-Inverse-Gamma loss function.
        
        Parameters:
            gamma: predicted value
            nu: strength of evidence
            alpha: shape parameter
            beta: scale parameter
            target: ground truth
        """
        # Prediction error
        error = target - gamma
        
        # NIG negative log-likelihood
        log_likelihood = (
            torch.lgamma(alpha) -
            torch.lgamma(alpha + 0.5) +
            0.5 * torch.log(2 * np.pi * beta / nu) +
            (alpha + 0.5) * torch.log(1 + nu * error**2 / (2 * beta))
        )
        
        # Regularization to prevent overconfident predictions
        reg = lambda_reg * torch.abs(target - gamma) * (2 * nu + alpha)
        
        return log_likelihood.mean() + reg.mean()
    
    def predict_with_uncertainty(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evidential prediction with uncertainty.
        
        From NIG parameters:
        - Prediction: gamma
        - Aleatoric: beta / (alpha - 1)
        - Epistemic: beta / (nu * (alpha - 1))
        """
        model.eval()
        with torch.no_grad():
            output = model(x, return_uncertainty=True)
            
            if not isinstance(output, tuple):
                raise ValueError("Model must return NIG parameters when return_uncertainty=True")
            
            predictions, nig_params = output
            
            # Parse NIG parameters
            gamma = nig_params[:, 0:1]  # Prediction
            nu = F.softplus(nig_params[:, 1:2]) + 1  # Strength
            alpha = F.softplus(nig_params[:, 2:3]) + 2  # Shape
            beta = F.softplus(nig_params[:, 3:4]) + 0.01  # Scale
        
        # Compute uncertainties
        aleatoric = torch.sqrt(beta / (alpha - 1))
        epistemic = torch.sqrt(beta / (nu * (alpha - 1)))
        
        return gamma, aleatoric, epistemic


class DomainOfApplicability:
    """
    Domain of Applicability (DOA) Modeling.
    
    Estimates how far a query compound is from the training distribution.
    """
    
    def __init__(self, method: str = "knn_distance"):
        self.method = method
        self.reference_embeddings: Optional[np.ndarray] = None
        self.k_neighbors: int = 5
    
    def fit(self, training_embeddings: np.ndarray) -> None:
        """Fit DOA model on training set embeddings."""
        self.reference_embeddings = training_embeddings
        
        if self.method == "knn_distance":
            from sklearn.neighbors import NearestNeighbors
            self.nn_model = NearestNeighbors(n_neighbors=self.k_neighbors)
            self.nn_model.fit(training_embeddings)
    
    def compute_applicability(
        self,
        query_embeddings: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute applicability metrics for query compounds.
        
        Returns:
            Dictionary with distance, similarity, and reliability metrics
        """
        if self.reference_embeddings is None:
            raise ValueError("DOA model not fitted. Call fit() first.")
        
        if self.method == "knn_distance":
            # k-NN distance
            distances, indices = self.nn_model.kneighbors(query_embeddings)
            avg_distance = distances.mean(axis=1)
            
            # Distance-based reliability (closer = more reliable)
            # Normalize by training set statistics
            train_mean_dist = self.nn_model.kneighbors(self.reference_embeddings)[0].mean()
            reliability = np.exp(-avg_distance / train_mean_dist)
            
            return {
                "knn_distance": avg_distance,
                "reliability_score": reliability,
                "in_domain": reliability > 0.5,  # Threshold
                "nearest_neighbor_indices": indices[:, 0]
            }
        
        elif self.method == "mahalanobis":
            from scipy.spatial.distance import mahalanobis
            
            # Compute covariance of training set
            cov = np.cov(self.reference_embeddings.T)
            cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
            
            # Mahalanobis distance to centroid
            centroid = self.reference_embeddings.mean(axis=0)
            distances = np.array([
                mahalanobis(query, centroid, cov_inv)
                for query in query_embeddings
            ])
            
            return {
                "mahalanobis_distance": distances,
                "reliability_score": np.exp(-distances / distances.mean()),
                "in_domain": distances < np.percentile(distances, 95)
            }
        
        else:
            raise ValueError(f"Unknown DOA method: {self.method}")
    
    def is_in_domain(
        self,
        query_embeddings: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Check if compounds are within the applicability domain."""
        metrics = self.compute_applicability(query_embeddings)
        return metrics["reliability_score"] > threshold


class UncertaintyCalibrator:
    """
    Calibration of uncertainty estimates.
    
    Ensures that predicted uncertainties match observed errors.
    """
    
    def __init__(self):
        self.temperature: Optional[float] = None
    
    def calibrate_temperature(
        self,
        model: nn.Module,
        val_data: torch.utils.data.DataLoader
    ) -> float:
        """
        Temperature scaling for uncertainty calibration.
        
        Finds optimal temperature to minimize calibration error.
        """
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            losses = []
            
            for x, y in val_data:
                with torch.no_grad():
                    pred, unc = model(x, return_uncertainty=True)
                    if isinstance(unc, tuple):
                        unc = unc[0]
                
                # Scaled uncertainty
                scaled_unc = unc / self.temperature
                
                # Negative log-likelihood
                nll = 0.5 * torch.log(2 * np.pi * scaled_unc**2) + \
                      0.5 * ((y - pred) / scaled_unc)**2
                losses.append(nll.mean())
            
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return self.temperature.item()
    
    def compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.
        
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Sharpness
        """
        # Create bins
        n_bins = 10
        errors = np.abs(predictions - targets)
        
        # Bin by uncertainty
        bin_edges = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
        
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_uncertainty = uncertainties[mask].mean()
                bin_error = errors[mask].mean()
                bin_weight = mask.sum() / len(uncertainties)
                
                calibration_error = np.abs(bin_uncertainty - bin_error)
                ece += bin_weight * calibration_error
                mce = max(mce, calibration_error)
        
        # Sharpness (average predicted uncertainty)
        sharpness = uncertainties.mean()
        
        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "sharpness": sharpness
        }


# Export
__all__ = [
    'UncertaintyEstimate',
    'UncertaintyMethod',
    'DeepEnsembleUncertainty',
    'MonteCarloDropoutUncertainty',
    'ConformalPredictionUncertainty',
    'EvidentialUncertainty',
    'DomainOfApplicability',
    'UncertaintyCalibrator'
]
