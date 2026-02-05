"""
VS-Bench - Continuous Benchmark Evolution
==========================================

Framework for benchmark versioning, hidden test sets,
contamination monitoring, and community-driven evolution.

Author: VS-Bench Development Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4
import pandas as pd


@dataclass
class DatasetVersion:
    """Version information for a dataset."""
    major: int
    minor: int
    patch: int
    release_date: datetime = field(default_factory=datetime.utcnow)
    changes: str = ""
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: DatasetVersion) -> bool:
        """Check if versions are compatible (same major)."""
        return self.major == other.major


@dataclass
class HiddenTestSet:
    """
    Hidden test set for unbiased evaluation.
    
    Maintains a separate test set that is not publicly
    available to prevent overfitting and data contamination.
    """
    
    test_set_id: UUID = field(default_factory=uuid4)
    name: str = ""
    version: DatasetVersion = field(default_factory=lambda: DatasetVersion(1, 0, 0))
    
    # Test set content
    num_samples: int = 0
    compound_hashes: List[str] = field(default_factory=list)  # Hashes only, not actual data
    target_hashes: List[str] = field(default_factory=list)
    
    # Access control
    authorized_evaluators: List[str] = field(default_factory=list)
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Refresh schedule
    created_at: datetime = field(default_factory=datetime.utcnow)
    refresh_date: Optional[datetime] = None
    
    def record_access(
        self,
        evaluator: str,
        submission_id: UUID,
        timestamp: datetime = None
    ) -> None:
        """Record access to hidden test set."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.access_log.append({
            "evaluator": evaluator,
            "submission_id": str(submission_id),
            "timestamp": timestamp.isoformat()
        })
    
    def should_refresh(self) -> bool:
        """Check if test set should be refreshed."""
        if self.refresh_date is None:
            return False
        return datetime.utcnow() >= self.refresh_date
    
    def to_public_info(self) -> Dict[str, Any]:
        """Get public information about test set (no actual data)."""
        return {
            "test_set_id": str(self.test_set_id),
            "name": self.name,
            "version": str(self.version),
            "num_samples": self.num_samples,
            "created_at": self.created_at.isoformat(),
            "next_refresh": self.refresh_date.isoformat() if self.refresh_date else None,
            "access_count": len(self.access_log)
        }


@dataclass
class ContaminationCheck:
    """Result of contamination check."""
    
    check_id: UUID = field(default_factory=uuid4)
    submission_id: UUID = field(default_factory=uuid4)
    
    # Check results
    is_contaminated: bool = False
    contamination_type: Optional[str] = None  # data_leakage, memorization, improper_preprocessing
    confidence: float = 0.0
    
    # Evidence
    suspicious_compounds: List[str] = field(default_factory=list)
    statistical_anomalies: List[str] = field(default_factory=list)
    
    # Action taken
    action: str = "none"  # none, warning, rejection, investigation
    checked_at: datetime = field(default_factory=datetime.utcnow)


class ContaminationMonitor:
    """
    Monitor for benchmark contamination and data leakage.
    
    Implements multiple strategies to detect:
    - Data leakage from test to training
    - Model memorization
    - Improper data preprocessing
    """
    
    def __init__(self):
        self.known_compounds: Set[str] = set()
        self.submission_history: List[UUID] = []
        self.contamination_checks: List[ContaminationCheck] = []
    
    def register_known_compounds(self, compound_hashes: List[str]) -> None:
        """Register compounds from training/hidden sets."""
        self.known_compounds.update(compound_hashes)
    
    def check_submission(
        self,
        submission_id: UUID,
        predictions_df: pd.DataFrame,
        training_compounds: Optional[Set[str]] = None
    ) -> ContaminationCheck:
        """
        Check submission for contamination.
        
        Args:
            submission_id: Submission UUID
            predictions_df: Prediction dataframe
            training_compounds: Set of compounds used in training
            
        Returns:
            ContaminationCheck result
        """
        check = ContaminationCheck(submission_id=submission_id)
        
        # Check 1: Perfect predictions on known compounds
        if len(predictions_df) > 0:
            # This would check if predictions are suspiciously perfect
            # on compounds that might have been in training data
            pass
        
        # Check 2: Statistical anomalies
        suspicious_patterns = self._detect_statistical_anomalies(predictions_df)
        if suspicious_patterns:
            check.statistical_anomalies = suspicious_patterns
        
        # Check 3: Compound overlap with hidden test
        if training_compounds:
            overlap = training_compounds & self.known_compounds
            if overlap:
                check.suspicious_compounds = list(overlap)
                check.is_contaminated = True
                check.contamination_type = "data_leakage"
                check.confidence = len(overlap) / len(predictions_df)
                check.action = "investigation" if check.confidence > 0.1 else "warning"
        
        self.contamination_checks.append(check)
        return check
    
    def _detect_statistical_anomalies(
        self,
        predictions_df: pd.DataFrame
    ) -> List[str]:
        """Detect statistical anomalies in predictions."""
        anomalies = []
        
        if "predicted_value" in predictions_df.columns:
            values = predictions_df["predicted_value"]
            
            # Check for uniform predictions
            if values.nunique() < 10:
                anomalies.append("Suspiciously few unique prediction values")
            
            # Check for perfect clustering
            if len(values) > 100:
                # Would implement more sophisticated checks here
                pass
        
        return anomalies
    
    def get_contamination_report(self) -> Dict[str, Any]:
        """Get contamination monitoring report."""
        contaminated = [c for c in self.contamination_checks if c.is_contaminated]
        
        return {
            "total_checks": len(self.contamination_checks),
            "contaminated": len(contaminated),
            "contamination_rate": len(contaminated) / len(self.contamination_checks) if self.contamination_checks else 0,
            "by_type": {}
        }


class BenchmarkEvolutionManager:
    """
    Manages continuous evolution of benchmark suite.
    
    Handles:
    - Dataset versioning
    - Hidden test set refresh
    - Community voting for new tasks
    - Contamination monitoring
    """
    
    def __init__(self, benchmark_suite, storage_path: str = "./evolution"):
        self.suite = benchmark_suite
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.hidden_test_sets: Dict[str, HiddenTestSet] = {}
        self.contamination_monitor = ContaminationMonitor()
        
        # Version tracking
        self.dataset_versions: Dict[str, DatasetVersion] = {}
        self.version_history: List[Dict[str, Any]] = []
        
        # Community input
        self.proposed_tasks: List[Dict[str, Any]] = []
        self.task_votes: Dict[str, Dict[str, int]] = {}
    
    def register_dataset_version(
        self,
        dataset_name: str,
        version: DatasetVersion,
        changes: str
    ) -> None:
        """Register a new dataset version."""
        self.dataset_versions[dataset_name] = version
        
        self.version_history.append({
            "dataset": dataset_name,
            "version": str(version),
            "changes": changes,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Save version info
        version_file = self.storage_path / f"{dataset_name}_version.json"
        version_file.write_text(json.dumps({
            "version": str(version),
            "changes": changes,
            "history": self.version_history
        }, indent=2))
    
    def create_hidden_test_set(
        self,
        name: str,
        num_samples: int,
        refresh_interval_days: int = 365
    ) -> HiddenTestSet:
        """
        Create a new hidden test set.
        
        Args:
            name: Test set name
            num_samples: Number of samples
            refresh_interval_days: Days until refresh
            
        Returns:
            HiddenTestSet object
        """
        test_set = HiddenTestSet(
            name=name,
            num_samples=num_samples,
            refresh_date=datetime.utcnow() + timedelta(days=refresh_interval_days)
        )
        
        self.hidden_test_sets[name] = test_set
        
        return test_set
    
    def refresh_test_set(self, name: str) -> HiddenTestSet:
        """
        Refresh a hidden test set with new data.
        
        This archives the old test set and creates a new one,
        ensuring continuous evaluation without staleness.
        
        Args:
            name: Test set to refresh
            
        Returns:
            New HiddenTestSet
        """
        if name not in self.hidden_test_sets:
            raise ValueError(f"Test set '{name}' not found")
        
        old_test_set = self.hidden_test_sets[name]
        
        # Archive old test set
        archive_path = self.storage_path / "archived_test_sets"
        archive_path.mkdir(exist_ok=True)
        
        archive_file = archive_path / f"{name}_{old_test_set.version}_{datetime.utcnow().strftime('%Y%m%d')}.json"
        archive_file.write_text(json.dumps(old_test_set.to_public_info(), indent=2))
        
        # Create new version
        new_version = DatasetVersion(
            major=old_test_set.version.major,
            minor=old_test_set.version.minor + 1,
            patch=0,
            changes=f"Annual refresh from version {old_test_set.version}"
        )
        
        new_test_set = HiddenTestSet(
            name=name,
            version=new_version,
            num_samples=old_test_set.num_samples,
            refresh_date=datetime.utcnow() + timedelta(days=365)
        )
        
        self.hidden_test_sets[name] = new_test_set
        
        return new_test_set
    
    def propose_new_task(
        self,
        proposer: str,
        task_name: str,
        description: str,
        motivation: str
    ) -> str:
        """
        Propose a new benchmark task.
        
        Args:
            proposer: GitHub username
            task_name: Task name
            description: Task description
            motivation: Why this task should be added
            
        Returns:
            Proposal ID
        """
        proposal_id = str(uuid4())[:8]
        
        proposal = {
            "id": proposal_id,
            "proposer": proposer,
            "task_name": task_name,
            "description": description,
            "motivation": motivation,
            "votes_for": 0,
            "votes_against": 0,
            "status": "proposed",
            "proposed_at": datetime.utcnow().isoformat()
        }
        
        self.proposed_tasks.append(proposal)
        self.task_votes[proposal_id] = {"for": 0, "against": 0}
        
        # Save proposal
        proposals_file = self.storage_path / "proposed_tasks.json"
        proposals_file.write_text(json.dumps(self.proposed_tasks, indent=2))
        
        return proposal_id
    
    def vote_for_task(
        self,
        proposal_id: str,
        voter: str,
        vote_for: bool
    ) -> bool:
        """
        Vote on a proposed task.
        
        Args:
            proposal_id: Proposal ID
            voter: Voter username
            vote_for: True for yes, False for no
            
        Returns:
            Success status
        """
        if proposal_id not in self.task_votes:
            return False
        
        key = "for" if vote_for else "against"
        self.task_votes[proposal_id][key] += 1
        
        # Update proposal
        for proposal in self.proposed_tasks:
            if proposal["id"] == proposal_id:
                proposal["votes_for"] = self.task_votes[proposal_id]["for"]
                proposal["votes_against"] = self.task_votes[proposal_id]["against"]
                
                # Check if accepted
                total_votes = proposal["votes_for"] + proposal["votes_against"]
                if total_votes >= 10:
                    score = proposal["votes_for"] / total_votes
                    if score >= 0.7:
                        proposal["status"] = "accepted"
                    elif score <= 0.3:
                        proposal["status"] = "rejected"
                break
        
        # Save updated proposals
        proposals_file = self.storage_path / "proposed_tasks.json"
        proposals_file.write_text(json.dumps(self.proposed_tasks, indent=2))
        
        return True
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        # Check for test sets needing refresh
        needs_refresh = [
            name for name, ts in self.hidden_test_sets.items()
            if ts.should_refresh()
        ]
        
        # Get pending proposals
        pending_proposals = [
            p for p in self.proposed_tasks
            if p["status"] == "proposed"
        ]
        
        return {
            "dataset_versions": {
                name: str(version)
                for name, version in self.dataset_versions.items()
            },
            "hidden_test_sets": {
                name: ts.to_public_info()
                for name, ts in self.hidden_test_sets.items()
            },
            "test_sets_needing_refresh": needs_refresh,
            "pending_proposals": len(pending_proposals),
            "total_proposals": len(self.proposed_tasks),
            "accepted_proposals": len([p for p in self.proposed_tasks if p["status"] == "accepted"]),
            "contamination_status": self.contamination_monitor.get_contamination_report()
        }


# Export
__all__ = [
    'DatasetVersion',
    'HiddenTestSet',
    'ContaminationCheck',
    'ContaminationMonitor',
    'BenchmarkEvolutionManager'
]
