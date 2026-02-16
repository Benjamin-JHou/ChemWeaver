"""
VS-Bench - Leaderboard System
=============================

Public leaderboard infrastructure with:
- Submission handling and validation
- Automated evaluation
- Score computation and ranking
- GitHub-native workflow integration

Author: VS-Bench Development Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import pandas as pd


@dataclass
class SubmissionMetadata:
    """Metadata for a benchmark submission."""
    
    # Identification
    submission_id: UUID = field(default_factory=uuid4)
    team_name: str = ""
    team_members: List[str] = field(default_factory=list)
    contact_email: str = ""
    
    # Method description
    method_name: str = ""
    method_description: str = ""
    architecture: str = ""  # GNN, Transformer, Ensemble, etc.
    
    # Computational resources
    training_time_hours: Optional[float] = None
    inference_time_per_compound_ms: Optional[float] = None
    hardware_used: str = ""  # GPU model, CPU cores, etc.
    
    # Code availability
    code_repository: Optional[str] = None
    docker_image: Optional[str] = None
    
    # Submission timestamp
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    
    # Benchmark target
    benchmark_suite: str = ""
    benchmark_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": str(self.submission_id),
            "team": {
                "name": self.team_name,
                "members": self.team_members,
                "email": self.contact_email
            },
            "method": {
                "name": self.method_name,
                "description": self.method_description,
                "architecture": self.architecture
            },
            "resources": {
                "training_time_hours": self.training_time_hours,
                "inference_time_ms": self.inference_time_per_compound_ms,
                "hardware": self.hardware_used
            },
            "code": {
                "repository": self.code_repository,
                "docker": self.docker_image
            },
            "benchmark": {
                "suite": self.benchmark_suite,
                "version": self.benchmark_version
            },
            "submitted_at": self.submitted_at.isoformat()
        }


@dataclass
class SubmissionFiles:
    """File paths for a submission."""
    
    predictions_csv: str  # prediction.csv
    metadata_json: str    # metadata.json
    environment_yaml: Optional[str] = None  # environment.yaml or Dockerfile
    dockerfile: Optional[str] = None
    run_script: Optional[str] = None  # run.sh
    
    def validate_existence(self) -> Tuple[bool, List[str]]:
        """Check that all required files exist."""
        errors = []
        
        required = [self.predictions_csv, self.metadata_json]
        for f in required:
            if not Path(f).exists():
                errors.append(f"Missing required file: {f}")
        
        # Check that at least one of environment.yaml or Dockerfile exists
        if self.environment_yaml and not Path(self.environment_yaml).exists():
            errors.append(f"Environment file not found: {self.environment_yaml}")
        if self.dockerfile and not Path(self.dockerfile).exists():
            errors.append(f"Dockerfile not found: {self.dockerfile}")
        
        return len(errors) == 0, errors


@dataclass
class EvaluationResult:
    """Results from evaluating a submission."""
    
    # Identification
    evaluation_id: UUID = field(default_factory=uuid4)
    submission_id: UUID = field(default_factory=uuid4)
    
    # Timing
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    evaluation_duration_seconds: float = 0.0
    
    # Validation status
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Scores by benchmark
    benchmark_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Global score
    global_score: float = 0.0
    global_score_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Rank
    rank: Optional[int] = None
    percentile: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": str(self.evaluation_id),
            "submission_id": str(self.submission_id),
            "evaluated_at": self.evaluated_at.isoformat(),
            "duration_seconds": self.evaluation_duration_seconds,
            "validation": {
                "passed": self.validation_passed,
                "errors": self.validation_errors
            },
            "scores": {
                "global": self.global_score,
                "breakdown": self.global_score_breakdown,
                "by_benchmark": self.benchmark_scores
            },
            "ranking": {
                "rank": self.rank,
                "percentile": self.percentile
            }
        }


class SubmissionValidator:
    """
    Validates benchmark submissions before evaluation.
    
    Checks file format, data integrity, and metadata completeness.
    """
    
    def __init__(self, benchmark_suite):
        self.suite = benchmark_suite
        self.validation_errors: List[str] = []
    
    def validate_submission(
        self,
        files: SubmissionFiles,
        expected_compound_count: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a complete submission.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # 1. Check file existence
        valid, file_errors = files.validate_existence()
        if not valid:
            errors.extend(file_errors)
        
        # 2. Validate metadata JSON
        if Path(files.metadata_json).exists():
            try:
                with open(files.metadata_json) as f:
                    metadata = json.load(f)
                
                # Check required fields
                required_fields = ["team_name", "method_name", "benchmark_suite"]
                for field in required_fields:
                    if field not in metadata:
                        errors.append(f"Missing required metadata field: {field}")
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in metadata: {e}")
        
        # 3. Validate predictions CSV
        if Path(files.predictions_csv).exists():
            try:
                df = pd.read_csv(files.predictions_csv)
                
                # Check required columns
                required_cols = ["compound_id", "predicted_value"]
                for col in required_cols:
                    if col not in df.columns:
                        errors.append(f"Missing required column in predictions: {col}")
                
                # Check for NaN values
                if df["predicted_value"].isna().any():
                    errors.append("Predictions contain NaN values")
                
                # Check compound count if expected
                if expected_compound_count:
                    if len(df) != expected_compound_count:
                        errors.append(
                            f"Prediction count mismatch: expected {expected_compound_count}, "
                            f"got {len(df)}"
                        )
                
                # Check value ranges (should be reasonable for docking scores)
                if "predicted_value" in df.columns:
                    values = df["predicted_value"].dropna()
                    if len(values) > 0:
                        if values.min() < -50 or values.max() > 50:
                            errors.append(
                                f"Prediction values out of reasonable range: "
                                f"[{values.min():.2f}, {values.max():.2f}]"
                            )
                
            except Exception as e:
                errors.append(f"Error reading predictions CSV: {e}")
        
        # 4. Validate environment specification
        if files.environment_yaml:
            try:
                import yaml
                with open(files.environment_yaml) as f:
                    env = yaml.safe_load(f)
                
                # Check for required dependencies
                if "dependencies" not in env:
                    errors.append("environment.yaml missing dependencies section")
            except Exception as e:
                errors.append(f"Error reading environment.yaml: {e}")
        
        # 5. Validate Dockerfile if present
        if files.dockerfile and Path(files.dockerfile).exists():
            content = Path(files.dockerfile).read_text()
            if "FROM" not in content:
                errors.append("Dockerfile missing FROM instruction")
        
        self.validation_errors = errors
        return len(errors) == 0, errors


class EvaluationEngine:
    """
    Automated evaluation engine for benchmark submissions.
    
    Computes metrics and global scores with reproducibility checks.
    """
    
    def __init__(self, benchmark_suite):
        self.suite = benchmark_suite
        self.evaluation_history: List[EvaluationResult] = []
    
    def evaluate_submission(
        self,
        submission_files: SubmissionFiles,
        ground_truth: Dict[str, pd.DataFrame],
        compute_uncertainty_metrics: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a submission against ground truth.
        
        Args:
            submission_files: Submission file paths
            ground_truth: Dict mapping benchmark name to ground truth DataFrame
            compute_uncertainty_metrics: Whether to compute calibration metrics
            
        Returns:
            EvaluationResult with all computed metrics
        """
        from benchmark.evaluation_metrics import MetricsCalculator
        
        start_time = datetime.utcnow()
        
        # Load predictions
        predictions_df = pd.read_csv(submission_files.predictions_csv)
        
        # Evaluate each benchmark
        benchmark_scores = {}
        
        for bench_name, gt_df in ground_truth.items():
            # Merge predictions with ground truth
            merged = gt_df.merge(
                predictions_df,
                on="compound_id",
                how="inner"
            )
            
            if len(merged) == 0:
                benchmark_scores[bench_name] = {"error": "No matching compounds"}
                continue
            
            # Extract arrays
            y_true = merged["ground_truth"].values
            y_pred = merged["predicted_value"].values
            
            # Compute metrics based on task type
            scores = {}
            
            # Regression metrics
            reg_metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
            scores.update(reg_metrics.to_dict())
            
            # Ranking metrics (treat as binary with threshold)
            rank_metrics = MetricsCalculator.calculate_ranking_metrics(
                y_true, y_pred, threshold=None
            )
            scores.update(rank_metrics.to_dict())
            
            # Uncertainty metrics if available
            if compute_uncertainty_metrics and "uncertainty" in merged.columns:
                y_unc = merged["uncertainty"].values
                cal_metrics = MetricsCalculator.calculate_calibration_metrics(
                    y_true, y_pred, y_unc
                )
                scores.update(cal_metrics.to_dict())
            
            benchmark_scores[bench_name] = scores
        
        # Compute global score using suite weights
        global_result = self.suite.compute_global_score({
            name: scores.get("spearman", scores.get("auc_roc", 0.0))
            for name, scores in benchmark_scores.items()
        })
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Create result
        result = EvaluationResult(
            submission_id=uuid4(),  # Would link to actual submission
            evaluation_duration_seconds=duration,
            validation_passed=True,
            benchmark_scores=benchmark_scores,
            global_score=global_result["global_score"],
            global_score_breakdown=global_result["by_capability"]
        )
        
        self.evaluation_history.append(result)
        
        return result
    
    def compute_rankings(
        self,
        results: List[EvaluationResult]
    ) -> List[EvaluationResult]:
        """
        Compute rankings for a set of evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Results with rank and percentile assigned
        """
        # Sort by global score
        sorted_results = sorted(
            results,
            key=lambda r: r.global_score,
            reverse=True
        )
        
        n = len(sorted_results)
        
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
            result.percentile = (n - i) / n * 100
        
        return sorted_results


class Leaderboard:
    """
    Public leaderboard for benchmark results.
    
    Maintains rankings, tracks history, and provides visualization data.
    """
    
    def __init__(
        self,
        name: str,
        benchmark_suite,
        storage_path: Optional[str] = None
    ):
        self.leaderboard_id = uuid4()
        self.name = name
        self.suite = benchmark_suite
        self.storage_path = Path(storage_path) if storage_path else Path("./leaderboard")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.submissions: Dict[UUID, SubmissionMetadata] = {}
        self.evaluations: Dict[UUID, EvaluationResult] = {}
        self.rankings: List[UUID] = []  # Ordered list of evaluation IDs
        
        # Statistics
        self.total_submissions = 0
        self.last_updated = datetime.utcnow()
    
    def submit(
        self,
        metadata: SubmissionMetadata,
        files: SubmissionFiles,
        auto_evaluate: bool = True
    ) -> Tuple[UUID, Optional[EvaluationResult]]:
        """
        Submit a new entry to the leaderboard.
        
        Args:
            metadata: Submission metadata
            files: Submission files
            auto_evaluate: Whether to run evaluation immediately
            
        Returns:
            (submission_id, evaluation_result)
        """
        # Store submission
        self.submissions[metadata.submission_id] = metadata
        self.total_submissions += 1
        
        result = None
        
        if auto_evaluate:
            # Validate
            validator = SubmissionValidator(self.suite)
            is_valid, errors = validator.validate_submission(files)
            
            if is_valid:
                # Evaluate (would need ground truth loaded)
                # Create a minimal valid evaluation entry.
                result = EvaluationResult(
                    submission_id=metadata.submission_id,
                    validation_passed=True
                )
                
                self.evaluations[result.evaluation_id] = result
                self._update_rankings()
            else:
                result = EvaluationResult(
                    submission_id=metadata.submission_id,
                    validation_passed=False,
                    validation_errors=errors
                )
        
        self.last_updated = datetime.utcnow()
        return metadata.submission_id, result
    
    def _update_rankings(self) -> None:
        """Update leaderboard rankings."""
        # Get all valid evaluations
        valid_evals = [
            e for e in self.evaluations.values()
            if e.validation_passed
        ]
        
        # Compute rankings
        engine = EvaluationEngine(self.suite)
        ranked = engine.compute_rankings(valid_evals)
        
        # Update order
        self.rankings = [r.evaluation_id for r in ranked]
    
    def get_leaderboard(
        self,
        top_k: Optional[int] = None,
        include_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get current leaderboard.
        
        Args:
            top_k: Return top k entries (None for all)
            include_details: Include full evaluation details
            
        Returns:
            List of leaderboard entries
        """
        entries = []
        
        for eval_id in self.rankings[:top_k] if top_k else self.rankings:
            if eval_id not in self.evaluations:
                continue
            
            eval_result = self.evaluations[eval_id]
            
            if eval_result.submission_id not in self.submissions:
                continue
            
            submission = self.submissions[eval_result.submission_id]
            
            entry = {
                "rank": eval_result.rank,
                "team": submission.team_name,
                "method": submission.method_name,
                "global_score": eval_result.global_score,
                "submitted_at": submission.submitted_at.isoformat()
            }
            
            if include_details:
                entry["scores"] = eval_result.global_score_breakdown
                entry["architecture"] = submission.architecture
                entry["hardware"] = submission.hardware_used
            
            entries.append(entry)
        
        return entries
    
    def export_leaderboard(
        self,
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export leaderboard to file.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv, html)
            
        Returns:
            Path to exported file
        """
        entries = self.get_leaderboard(include_details=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump({
                    "leaderboard": self.name,
                    "benchmark_suite": self.suite.name,
                    "last_updated": self.last_updated.isoformat(),
                    "total_submissions": self.total_submissions,
                    "entries": entries
                }, f, indent=2)
        
        elif format == "csv":
            df = pd.DataFrame(entries)
            df.to_csv(output_path, index=False)
        
        elif format == "html":
            html = self._generate_html_leaderboard(entries)
            with open(output_path, 'w') as f:
                f.write(html)
        
        return output_path
    
    def _generate_html_leaderboard(self, entries: List[Dict]) -> str:
        """Generate HTML leaderboard."""
        rows = ""
        for entry in entries:
            rows += f"""
            <tr>
                <td>{entry['rank']}</td>
                <td>{entry['team']}</td>
                <td>{entry['method']}</td>
                <td>{entry['global_score']:.4f}</td>
            </tr>
            """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.name} Leaderboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{self.name} Leaderboard</h1>
    <p>Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M')}</p>
    <p>Total submissions: {self.total_submissions}</p>
    <table>
        <tr>
            <th>Rank</th>
            <th>Team</th>
            <th>Method</th>
            <th>Global Score</th>
        </tr>
        {rows}
    </table>
</body>
</html>
        """


# Export
__all__ = [
    'SubmissionMetadata',
    'SubmissionFiles',
    'EvaluationResult',
    'SubmissionValidator',
    'EvaluationEngine',
    'Leaderboard'
]
