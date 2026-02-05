"""
VS-Bench - Evaluation Metrics
=============================

Comprehensive evaluation metrics for virtual screening benchmarks:
- Regression metrics
- Ranking metrics
- Calibration metrics
- Robustness metrics

Author: VS-Bench Development Team
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)


@dataclass
class RegressionMetrics:
    """Regression evaluation metrics."""
    
    # Primary metrics
    rmse: float
    mae: float
    spearman: float
    pearson: float
    r2: float
    
    # Advanced metrics
    concordance_index: float
    kendall_tau: float
    
    # Top-k performance
    top_10_recall: float
    top_100_recall: float
    top_1000_recall: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "r2": self.r2,
            "concordance_index": self.concordance_index,
            "kendall_tau": self.kendall_tau,
            "top_10_recall": self.top_10_recall,
            "top_100_recall": self.top_100_recall,
            "top_1000_recall": self.top_1000_recall
        }


@dataclass
class RankingMetrics:
    """Ranking and enrichment metrics."""
    
    # Standard classification
    auc_roc: float
    auc_pr: float
    accuracy: float
    balanced_accuracy: float
    f1_score: float
    mcc: float  # Matthews Correlation Coefficient
    
    # Virtual screening specific
    bedroc: float  # Boltzmann-Enhanced Discrimination of ROC
    enrichment_factor_1: float
    enrichment_factor_5: float
    enrichment_factor_10: float
    
    # Precision at K
    precision_at_10: float
    precision_at_100: float
    precision_at_1000: float
    
    # Hit rates
    hit_rate_1: float
    hit_rate_5: float
    hit_rate_10: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_score": self.f1_score,
            "mcc": self.mcc,
            "bedroc": self.bedroc,
            "enrichment_factor_1": self.enrichment_factor_1,
            "enrichment_factor_5": self.enrichment_factor_5,
            "enrichment_factor_10": self.enrichment_factor_10,
            "precision_at_10": self.precision_at_10,
            "precision_at_100": self.precision_at_100,
            "precision_at_1000": self.precision_at_1000,
            "hit_rate_1": self.hit_rate_1,
            "hit_rate_5": self.hit_rate_5,
            "hit_rate_10": self.hit_rate_10
        }


@dataclass
class CalibrationMetrics:
    """Uncertainty calibration metrics."""
    
    # Expected Calibration Error
    ece: float  # Expected Calibration Error
    ace: float  # Adaptive Calibration Error
    
    # Brier score
    brier_score: float
    
    # Reliability diagram metrics
    reliability_slope: float
    reliability_intercept: float
    
    # Uncertainty metrics
    uncertainty_spread: float
    prediction_interval_coverage: float
    
    # Calibration curve
    calibration_bins: List[float]
    calibration_accuracies: List[float]
    calibration_confidences: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": self.ece,
            "ace": self.ace,
            "brier_score": self.brier_score,
            "reliability": {
                "slope": self.reliability_slope,
                "intercept": self.reliability_intercept
            },
            "uncertainty": {
                "spread": self.uncertainty_spread,
                "interval_coverage": self.prediction_interval_coverage
            }
        }


@dataclass
class RobustnessMetrics:
    """Robustness and generalization metrics."""
    
    # OOD generalization
    ood_performance_ratio: float
    ood_auc_drop: float
    
    # Cross-target transfer
    cross_target_score: float
    transfer_efficiency: float
    negative_transfer_instances: int
    
    # Uncertainty-error correlation
    uncertainty_error_correlation: float
    uncertainty_auroc: float  # AUROC using uncertainty for error detection
    
    # Distribution shift
    domain_gap: float
    distribution_distance: float
    
    # Consistency
    prediction_stability: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ood": {
                "performance_ratio": self.ood_performance_ratio,
                "auc_drop": self.ood_auc_drop
            },
            "transfer": {
                "cross_target_score": self.cross_target_score,
                "efficiency": self.transfer_efficiency,
                "negative_transfer": self.negative_transfer_instances
            },
            "uncertainty": {
                "error_correlation": self.uncertainty_error_correlation,
                "auroc": self.uncertainty_auroc
            },
            "distribution": {
                "domain_gap": self.domain_gap,
                "distance": self.distribution_distance
            },
            "prediction_stability": self.prediction_stability
        }


class MetricsCalculator:
    """
    Comprehensive metrics calculator for virtual screening.
    
    Computes regression, ranking, calibration, and robustness metrics
    with proper handling of edge cases and confidence intervals.
    """
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k_values: List[int] = None
    ) -> RegressionMetrics:
        """
        Calculate regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            k_values: Top-k values for recall calculation
            
        Returns:
            RegressionMetrics object
        """
        if k_values is None:
            k_values = [10, 100, 1000]
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Correlation metrics
        spearman, _ = stats.spearmanr(y_true, y_pred)
        pearson, _ = stats.pearsonr(y_true, y_pred)
        kendall, _ = stats.kendalltau(y_true, y_pred)
        
        # Concordance index (C-index)
        c_index = MetricsCalculator._concordance_index(y_true, y_pred)
        
        # Top-k recall
        top_k_recalls = {}
        for k in k_values:
            if k <= len(y_true):
                top_k_recalls[k] = MetricsCalculator._top_k_recall(y_true, y_pred, k)
        
        return RegressionMetrics(
            rmse=rmse,
            mae=mae,
            spearman=spearman if not np.isnan(spearman) else 0.0,
            pearson=pearson if not np.isnan(pearson) else 0.0,
            r2=r2,
            concordance_index=c_index,
            kendall_tau=kendall if not np.isnan(kendall) else 0.0,
            top_10_recall=top_k_recalls.get(10, 0.0),
            top_100_recall=top_k_recalls.get(100, 0.0),
            top_1000_recall=top_k_recalls.get(1000, 0.0)
        )
    
    @staticmethod
    def calculate_ranking_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: Optional[float] = None,
        enrichment_fractions: List[float] = None
    ) -> RankingMetrics:
        """
        Calculate ranking and classification metrics.
        
        Args:
            y_true: Ground truth labels (binary or continuous)
            y_score: Prediction scores
            threshold: Threshold for binary classification
            enrichment_fractions: Fractions for EF calculation
            
        Returns:
            RankingMetrics object
        """
        if enrichment_fractions is None:
            enrichment_fractions = [0.01, 0.05, 0.10]
        
        # Convert to binary if needed
        if threshold is not None:
            y_binary = (y_true <= threshold).astype(int)  # Active = low score (docking)
        else:
            y_binary = (y_true > np.median(y_true)).astype(int)
        
        # Standard metrics
        try:
            auc_roc = roc_auc_score(y_binary, -y_score)  # Negate because lower is better for docking
        except:
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(y_binary, -y_score)
        except:
            auc_pr = 0.0
        
        # BEDROC
        bedroc = MetricsCalculator._calculate_bedroc(y_binary, y_score)
        
        # Enrichment factors
        efs = {}
        for frac in enrichment_fractions:
            efs[int(frac * 100)] = MetricsCalculator._enrichment_factor(
                y_binary, y_score, frac
            )
        
        # Precision at K
        n_actives = y_binary.sum()
        precisions = {}
        for k in [10, 100, 1000]:
            if k <= len(y_binary):
                precisions[k] = MetricsCalculator._precision_at_k(y_binary, y_score, k)
        
        # Hit rates
        hit_rates = {}
        for frac in [0.01, 0.05, 0.10]:
            k = int(frac * len(y_binary))
            if k > 0:
                top_k_preds = (y_score.argsort()[:k] < n_actives).mean()
                hit_rates[int(frac * 100)] = float(top_k_preds)
        
        # Classification metrics with threshold
        y_pred_binary = (y_score <= np.percentile(y_score, 100 * (n_actives / len(y_binary)))).astype(int)
        
        accuracy = (y_pred_binary == y_binary).mean()
        balanced_accuracy = 0.5 * (
            (y_pred_binary[y_binary == 1] == 1).mean() +
            (y_pred_binary[y_binary == 0] == 0).mean()
        )
        
        # MCC
        tp = ((y_pred_binary == 1) & (y_binary == 1)).sum()
        tn = ((y_pred_binary == 0) & (y_binary == 0)).sum()
        fp = ((y_pred_binary == 1) & (y_binary == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_binary == 1)).sum()
        
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        
        # F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return RankingMetrics(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            f1_score=f1,
            mcc=mcc,
            bedroc=bedroc,
            enrichment_factor_1=efs.get(1, 1.0),
            enrichment_factor_5=efs.get(5, 1.0),
            enrichment_factor_10=efs.get(10, 1.0),
            precision_at_10=precisions.get(10, 0.0),
            precision_at_100=precisions.get(100, 0.0),
            precision_at_1000=precisions.get(1000, 0.0),
            hit_rate_1=hit_rates.get(1, 0.0),
            hit_rate_5=hit_rates.get(5, 0.0),
            hit_rate_10=hit_rates.get(10, 0.0)
        )
    
    @staticmethod
    def calculate_calibration_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_uncertainty: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> CalibrationMetrics:
        """
        Calculate calibration metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_uncertainty: Predictive uncertainties
            n_bins: Number of bins for ECE
            
        Returns:
            CalibrationMetrics object
        """
        # Reliability diagram data
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # For regression, we need to convert to confidence intervals
        if y_uncertainty is not None:
            # Assume Gaussian uncertainty
            coverage = np.abs(y_true - y_pred) <= (1.96 * y_uncertainty)
            confidences = 1 - np.exp(-0.5 * ((y_true - y_pred) / (y_uncertainty + 1e-8)) ** 2)
        else:
            confidences = np.ones_like(y_pred) * 0.5
            coverage = np.zeros_like(y_pred, dtype=bool)
        
        # Calculate ECE
        ece = 0.0
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = coverage[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
        
        # Brier score (for binary calibration)
        brier = brier_score_loss(coverage, confidences) if len(coverage) > 0 else 0.0
        
        # Reliability slope and intercept
        if len(bin_confidences) > 1:
            slope, intercept, _, _, _ = stats.linregress(bin_confidences, bin_accuracies)
        else:
            slope, intercept = 1.0, 0.0
        
        return CalibrationMetrics(
            ece=ece,
            ace=ece,  # Simplified - same as ECE
            brier_score=brier,
            reliability_slope=slope,
            reliability_intercept=intercept,
            uncertainty_spread=y_uncertainty.mean() if y_uncertainty is not None else 0.0,
            prediction_interval_coverage=coverage.mean() if len(coverage) > 0 else 0.0,
            calibration_bins=bin_boundaries.tolist(),
            calibration_accuracies=bin_accuracies,
            calibration_confidences=bin_confidences
        )
    
    @staticmethod
    def calculate_robustness_metrics(
        in_distribution_scores: Dict[str, float],
        out_distribution_scores: Dict[str, float],
        uncertainties: Optional[np.ndarray] = None,
        errors: Optional[np.ndarray] = None
    ) -> RobustnessMetrics:
        """
        Calculate robustness and generalization metrics.
        
        Args:
            in_distribution_scores: Metrics on in-distribution data
            out_distribution_scores: Metrics on OOD data
            uncertainties: Uncertainty estimates
            errors: Prediction errors
            
        Returns:
            RobustnessMetrics object
        """
        # OOD performance drop
        in_auc = in_distribution_scores.get("auc_roc", 0.5)
        ood_auc = out_distribution_scores.get("auc_roc", 0.5)
        ood_drop = in_auc - ood_auc
        ood_ratio = ood_auc / in_auc if in_auc > 0 else 0.0
        
        # Uncertainty-error correlation
        if uncertainties is not None and errors is not None:
            corr, _ = stats.pearsonr(uncertainties, np.abs(errors))
            
            # AUROC for error detection using uncertainty
            try:
                high_error = (np.abs(errors) > np.percentile(np.abs(errors), 90)).astype(int)
                uncertainty_auroc = roc_auc_score(high_error, uncertainties)
            except:
                uncertainty_auroc = 0.5
        else:
            corr = 0.0
            uncertainty_auroc = 0.5
        
        return RobustnessMetrics(
            ood_performance_ratio=ood_ratio,
            ood_auc_drop=ood_drop,
            cross_target_score=0.0,  # Would need cross-target data
            transfer_efficiency=0.0,
            negative_transfer_instances=0,
            uncertainty_error_correlation=corr if not np.isnan(corr) else 0.0,
            uncertainty_auroc=uncertainty_auroc,
            domain_gap=ood_drop,
            distribution_distance=0.0,  # Would need distribution metrics
            prediction_stability=1.0  # Would need multiple runs
        )
    
    @staticmethod
    def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate concordance index (C-index)."""
        n = len(y_true)
        concordant = 0
        permissible = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] != y_true[j]:
                    permissible += 1
                    if (y_true[i] < y_true[j]) == (y_pred[i] < y_pred[j]):
                        concordant += 1
        
        return concordant / permissible if permissible > 0 else 0.5
    
    @staticmethod
    def _top_k_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate top-k recall."""
        top_k_true = set(y_true.argsort()[:k])
        top_k_pred = set(y_pred.argsort()[:k])
        
        return len(top_k_true & top_k_pred) / k
    
    @staticmethod
    def _calculate_bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 80.0) -> float:
        """Calculate BEDROC score."""
        n_actives = y_true.sum()
        n_total = len(y_true)
        
        if n_actives == 0 or n_actives == n_total:
            return 0.0
        
        # Sort by score
        order = y_score.argsort()
        y_sorted = y_true[order]
        
        # Calculate BEDROC
        ri = np.where(y_sorted == 1)[0] + 1
        sum_exp = np.sum(np.exp(-alpha * ri / n_total))
        
        ra = n_actives / n_total
        bedroc = (
            sum_exp / (n_actives * (1 - np.exp(-alpha * ra)) / (n_total * (1 - np.exp(-alpha / n_total))))
            if n_actives > 0 else 0.0
        )
        
        return float(bedroc)
    
    @staticmethod
    def _enrichment_factor(
        y_true: np.ndarray,
        y_score: np.ndarray,
        fraction: float
    ) -> float:
        """Calculate enrichment factor at given fraction."""
        n_total = len(y_true)
        n_actives = y_true.sum()
        
        if n_actives == 0:
            return 1.0
        
        # Sort by score (ascending for docking)
        k = int(fraction * n_total)
        top_k_indices = y_score.argsort()[:k]
        
        # Count actives in top fraction
        n_actives_in_fraction = y_true[top_k_indices].sum()
        
        # EF = (% actives in top fraction) / (% actives overall)
        ef = (n_actives_in_fraction / k) / (n_actives / n_total) if k > 0 else 1.0
        
        return float(ef)
    
    @staticmethod
    def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        """Calculate precision at k."""
        top_k_indices = y_score.argsort()[:k]
        return float(y_true[top_k_indices].mean())


# Export
__all__ = [
    'RegressionMetrics',
    'RankingMetrics',
    'CalibrationMetrics',
    'RobustnessMetrics',
    'MetricsCalculator'
]
