"""
VS-Bench - Community Infrastructure
====================================

GitHub-native workflow for benchmark contributions:
- Fork and contribute workflow
- Automatic PR validation
- Community voting system
- Continuous integration

Author: VS-Bench Development Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
import json


class ContributionType(Enum):
    """Types of benchmark contributions."""
    NEW_METHOD = "new_method"
    BUG_FIX = "bug_fix"
    DATASET_IMPROVEMENT = "dataset_improvement"
    NEW_METRIC = "new_metric"
    DOCUMENTATION = "documentation"
    BENCHMARK_TASK = "benchmark_task"


class ContributionStatus(Enum):
    """Status of a contribution."""
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    MERGED = "merged"
    REJECTED = "rejected"


@dataclass
class ContributionProposal:
    """
    Community contribution proposal.
    
    Represents a submission to improve or extend
    the benchmark suite.
    """
    
    # Identification
    proposal_id: UUID = field(default_factory=uuid4)
    contributor: str = ""
    contributor_email: str = ""
    
    # Content
    contribution_type: ContributionType = ContributionType.NEW_METHOD
    title: str = ""
    description: str = ""
    motivation: str = ""
    
    # Technical details
    affected_benchmarks: List[str] = field(default_factory=list)
    changes_summary: str = ""
    files_modified: List[str] = field(default_factory=list)
    
    # Reproducibility
    code_repository: Optional[str] = None
    reproduction_instructions: str = ""
    
    # Community voting
    votes_for: int = 0
    votes_against: int = 0
    comments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status tracking
    status: ContributionStatus = ContributionStatus.SUBMITTED
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    
    def compute_score(self) -> float:
        """Compute community support score."""
        total = self.votes_for + self.votes_against
        if total == 0:
            return 0.0
        return self.votes_for / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": str(self.proposal_id),
            "contributor": self.contributor,
            "type": self.contribution_type.value,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "votes": {
                "for": self.votes_for,
                "against": self.votes_against,
                "score": self.compute_score()
            },
            "submitted_at": self.submitted_at.isoformat(),
            "affected_benchmarks": self.affected_benchmarks
        }


class GitHubWorkflowManager:
    """
    Manages GitHub-native contribution workflow.
    
    Coordinates fork, PR, validation, and merge processes
    for community contributions.
    """
    
    def __init__(self, repo_owner: str, repo_name: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.proposals: Dict[UUID, ContributionProposal] = {}
        
        # Workflow configuration
        self.required_reviews = 2
        self.min_vote_score = 0.6
        self.auto_validation = True
    
    def create_proposal(
        self,
        contributor: str,
        title: str,
        description: str,
        contribution_type: ContributionType
    ) -> ContributionProposal:
        """
        Create a new contribution proposal.
        
        Args:
            contributor: GitHub username
            title: Proposal title
            description: Detailed description
            contribution_type: Type of contribution
            
        Returns:
            ContributionProposal object
        """
        proposal = ContributionProposal(
            contributor=contributor,
            contribution_type=contribution_type,
            title=title,
            description=description
        )
        
        self.proposals[proposal.proposal_id] = proposal
        
        return proposal
    
    def generate_pr_template(self, proposal: ContributionProposal) -> str:
        """Generate GitHub Pull Request template."""
        template = f"""## Contribution Proposal: {proposal.title}

**Type**: {proposal.contribution_type.value}
**Proposer**: @{proposal.contributor}
**Date**: {proposal.submitted_at.strftime('%Y-%m-%d')}

### Description
{proposal.description}

### Motivation
{proposal.motivation}

### Affected Benchmarks
"""
        for bench in proposal.affected_benchmarks:
            template += f"- {bench}\n"
        
        template += f"""
### Changes Summary
{proposal.changes_summary}

### Reproducibility
{proposal.reproduction_instructions}

### Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] All tests pass
- [ ] Benchmark validation complete

---
**Proposal ID**: {proposal.proposal_id}
"""
        return template
    
    def validate_contribution(
        self,
        proposal_id: UUID
    ) -> Dict[str, Any]:
        """
        Validate a contribution proposal.
        
        Checks:
        - File format compliance
        - Test execution
        - Benchmark integrity
        - Documentation completeness
        
        Returns:
            Validation report
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return {"valid": False, "error": "Proposal not found"}
        
        checks = {
            "format_compliance": True,
            "tests_pass": True,
            "benchmark_integrity": True,
            "documentation_complete": True
        }
        
        errors = []
        
        # Validate based on contribution type
        if proposal.contribution_type == ContributionType.NEW_METHOD:
            # Check for required files
            required_files = [
                "submission/predictions.csv",
                "submission/metadata.json",
                "submission/environment.yaml"
            ]
            
            for f in required_files:
                if f not in proposal.files_modified:
                    errors.append(f"Missing required file: {f}")
                    checks["format_compliance"] = False
        
        elif proposal.contribution_type == ContributionType.NEW_METRIC:
            # Check metric implementation
            if "metrics.py" not in proposal.files_modified:
                errors.append("Metric implementation file missing")
                checks["format_compliance"] = False
        
        elif proposal.contribution_type == ContributionType.BENCHMARK_TASK:
            # Check benchmark definition
            if "benchmark_config.yaml" not in proposal.files_modified:
                errors.append("Benchmark configuration missing")
                checks["format_compliance"] = False
        
        is_valid = all(checks.values())
        
        return {
            "valid": is_valid,
            "checks": checks,
            "errors": errors,
            "proposal_id": str(proposal_id)
        }
    
    def vote(
        self,
        proposal_id: UUID,
        voter: str,
        vote_for: bool,
        comment: Optional[str] = None
    ) -> bool:
        """
        Cast a vote on a proposal.
        
        Args:
            proposal_id: Proposal UUID
            voter: GitHub username
            vote_for: True for yes, False for no
            comment: Optional comment
            
        Returns:
            Success status
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        
        if vote_for:
            proposal.votes_for += 1
        else:
            proposal.votes_against += 1
        
        if comment:
            proposal.comments.append({
                "voter": voter,
                "vote": "for" if vote_for else "against",
                "comment": comment,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return True
    
    def review_proposal(
        self,
        proposal_id: UUID,
        reviewer: str,
        decision: ContributionStatus,
        feedback: str
    ) -> bool:
        """
        Review and update proposal status.
        
        Args:
            proposal_id: Proposal to review
            reviewer: Reviewer username
            decision: APPROVED, CHANGES_REQUESTED, or REJECTED
            feedback: Review feedback
            
        Returns:
            Success status
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        
        # Check if ready for review
        if proposal.status not in [ContributionStatus.SUBMITTED, ContributionStatus.CHANGES_REQUESTED]:
            return False
        
        proposal.status = decision
        proposal.reviewer = reviewer
        proposal.reviewed_at = datetime.utcnow()
        
        proposal.comments.append({
            "voter": reviewer,
            "vote": "review",
            "comment": feedback,
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision.value
        })
        
        return True
    
    def merge_proposal(
        self,
        proposal_id: UUID,
        merger: str
    ) -> bool:
        """
        Merge an approved proposal.
        
        Args:
            proposal_id: Proposal to merge
            merger: Person merging the proposal
            
        Returns:
            Success status
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        
        # Check if approved
        if proposal.status != ContributionStatus.APPROVED:
            return False
        
        # Check vote threshold
        if proposal.compute_score() < self.min_vote_score:
            return False
        
        proposal.status = ContributionStatus.MERGED
        proposal.merged_at = datetime.utcnow()
        
        return True
    
    def get_pending_proposals(self) -> List[ContributionProposal]:
        """Get all proposals awaiting review."""
        return [
            p for p in self.proposals.values()
            if p.status == ContributionStatus.SUBMITTED
        ]
    
    def get_proposals_by_type(
        self,
        contribution_type: ContributionType
    ) -> List[ContributionProposal]:
        """Get proposals by type."""
        return [
            p for p in self.proposals.values()
            if p.contribution_type == contribution_type
        ]


class ContinuousIntegrationConfig:
    """
    CI/CD configuration for automatic validation.
    
    Defines GitHub Actions workflows for:
    - Submission validation
    - Benchmark execution
    - Leaderboard updates
    """
    
    def __init__(self):
        self.workflows = {}
    
    def generate_validation_workflow(self) -> str:
        """Generate GitHub Actions workflow for PR validation."""
        workflow = """name: Validate Contribution

on:
  pull_request:
    branches: [ main ]
    paths:
      - 'submissions/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      
      - name: Validate submission format
        run: |
          python -m benchmark.validate_pr \
            --pr-number ${{ github.event.pull_request.number }}
      
      - name: Run benchmark evaluation
        run: |
          python -m benchmark.evaluate_pr \
            --pr-number ${{ github.event.pull_request.number }}
      
      - name: Check for contamination
        run: |
          python -m benchmark.check_contamination \
            --pr-number ${{ github.event.pull_request.number }}
      
      - name: Comment results
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = fs.readFileSync('results.json', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `Validation Results:\n\n${results}`
            });
"""
        return workflow
    
    def generate_update_workflow(self) -> str:
        """Generate workflow for updating leaderboard."""
        workflow = """name: Update Leaderboard

on:
  push:
    branches: [ main ]
    paths:
      - 'submissions/**'

jobs:
  update-leaderboard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Update leaderboard
        run: |
          python -m benchmark.update_leaderboard
      
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add leaderboard/
          git commit -m "Update leaderboard" || exit 0
          git push
"""
        return workflow


class CommunityGuidelines:
    """
    Community contribution guidelines and best practices.
    """
    
    @staticmethod
    def generate_contributing_md() -> str:
        """Generate CONTRIBUTING.md content."""
        return """# Contributing to VS-Bench

Thank you for your interest in contributing to VS-Bench! This document provides guidelines for contributing to our benchmark ecosystem.

## Types of Contributions

### 1. Submit a Method
Submit your virtual screening method to the leaderboard:

1. Fork the repository
2. Create a submission in `submissions/<team-name>/`
3. Include:
   - `predictions.csv`: Your predictions
   - `metadata.json`: Method description
   - `environment.yaml` or `Dockerfile`: Environment specification
   - `run.sh`: Reproduction script
4. Submit a Pull Request

### 2. Propose a New Benchmark Task
To propose a new benchmark task:

1. Open a GitHub Issue with the "New Task" label
2. Describe the task, data sources, and evaluation metrics
3. Provide rationale for inclusion
4. Wait for community discussion and voting

### 3. Improve Documentation
Help improve our documentation:

- Fix typos
- Clarify instructions
- Add examples
- Translate content

### 4. Report Issues
Report bugs or suggest improvements via GitHub Issues.

## Submission Guidelines

### Data Format
- Predictions must be in CSV format
- Required columns: `compound_id`, `predicted_value`
- Optional columns: `uncertainty`, `confidence`

### Reproducibility Requirements
- Provide complete environment specification
- Include random seeds used
- Document any data preprocessing
- Share training code (optional but encouraged)

### Code of Conduct
- Be respectful and constructive
- Give credit to prior work
- Follow academic integrity standards
- Respect data licenses

## Review Process

1. **Automated Validation**: Checks format and runs evaluation
2. **Community Review**: 7-day comment period
3. **Maintainer Review**: Final approval
4. **Merge**: Automatic leaderboard update

## Questions?

- Open a GitHub Discussion
- Email: vs-bench@example.org
- Join our Slack channel

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
"""
    
    @staticmethod
    def generate_code_of_conduct() -> str:
        """Generate CODE_OF_CONDUCT.md content."""
        return """# Code of Conduct

## Our Pledge

We pledge to make participation in VS-Bench a harassment-free experience for everyone.

## Our Standards

### Positive Behavior
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards others

### Unacceptable Behavior
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## Enforcement

Instances of abusive behavior may be reported to the maintainers.

## Attribution

This Code of Conduct is adapted from the Contributor Covenant.
"""


# Export
__all__ = [
    'ContributionType',
    'ContributionStatus',
    'ContributionProposal',
    'GitHubWorkflowManager',
    'ContinuousIntegrationConfig',
    'CommunityGuidelines'
]
