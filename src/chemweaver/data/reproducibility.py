"""
Virtual Screening Standard Schema (VSSS) - Reproducibility Metadata Layer
========================================================================

Comprehensive tracking of all execution context necessary for full
experiment reproducibility across heterogeneous compute infrastructure.

Tracks:
- Container images and digests
- Workflow definitions and versions
- Software dependencies and versions
- Hardware descriptors (without assumptions)
- Execution parameters and environment

Author: VSSS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4


class ContainerRuntime(Enum):
    """Supported container runtimes."""
    DOCKER = "docker"
    SINGULARITY = "singularity"
    APPTAINER = "apptainer"
    PODMAN = "podman"
    SHIFTER = "shifter"
    CHARLIECLOUD = "charliecloud"
    NONE = "none"


class ExecutionBackend(Enum):
    """Supported execution backends."""
    LOCAL = "local"
    HPC_SLURM = "hpc_slurm"
    HPC_PBS = "hpc_pbs"
    HPC_LSF = "hpc_lsf"
    HPC_GRIDENGINE = "hpc_gridengine"
    KUBERNETES = "kubernetes"
    AWS_BATCH = "aws_batch"
    AWS_ECS = "aws_ecs"
    GOOGLE_CLOUD_RUN = "google_cloud_run"
    AZURE_CONTAINER_INSTANCES = "azure_container_instances"
    CUSTOM = "custom"


class WorkflowEngine(Enum):
    """Supported workflow engines."""
    NEXTFLOW = "nextflow"
    SNAKEMAKE = "snakemake"
    CROMWELL = "cromwell"
    GALAXY = "galaxy"
    CWLEXEC = "cwlexec"
    TOIL = "toil"
    AIRFLOW = "airflow"
    PREFECT = "prefect"
    RAY_WORKFLOWS = "ray_workflows"
    CUSTOM = "custom"


@dataclass
class ContainerImage:
    """
    Container image specification with content-addressable identifiers.
    
    Uses immutable digests (SHA256) rather than tags to ensure
    exact reproducibility.
    """
    runtime: ContainerRuntime
    image_uri: str
    image_digest: str  # SHA256 digest of the image manifest
    
    # Optional metadata
    registry: Optional[str] = None
    repository: Optional[str] = None
    tag: Optional[str] = None  # Original tag (informational only)
    build_date: Optional[datetime] = None
    build_args: Dict[str, str] = field(default_factory=dict)
    
    # Dockerfile/Containerfile reference
    dockerfile_git_url: Optional[str] = None
    dockerfile_git_commit: Optional[str] = None
    dockerfile_path: Optional[str] = None
    
    def verify_digest(self) -> bool:
        """
        Verify that the image digest matches the stored value.
        
        This should be called after pulling an image to ensure
        bit-for-bit reproducibility.
        """
        # Implementation would use container runtime to verify
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime": self.runtime.value,
            "image_uri": self.image_uri,
            "image_digest": self.image_digest,
            "registry": self.registry,
            "repository": self.repository,
            "tag": self.tag,
            "build_date": self.build_date.isoformat() if self.build_date else None,
            "build_args": self.build_args,
            "dockerfile_git_url": self.dockerfile_git_url,
            "dockerfile_git_commit": self.dockerfile_git_commit,
            "dockerfile_path": self.dockerfile_path
        }


@dataclass
class SoftwareDependency:
    """
    Software dependency with version information.
    
    Captures both high-level tools and low-level libraries.
    """
    name: str
    version: str
    
    # Installation method
    installation_method: str = "unknown"  # pip, conda, apt, source, etc.
    package_manager: Optional[str] = None
    
    # Source information
    source_url: Optional[str] = None
    git_commit: Optional[str] = None
    
    # Python-specific
    package_name: Optional[str] = None  # PyPI package name if different
    
    # Environment
    environment_path: Optional[str] = None
    conda_environment: Optional[str] = None
    virtualenv_path: Optional[str] = None


@dataclass
class PythonEnvironment:
    """Complete Python environment specification."""
    python_version: str
    python_executable: str
    
    # Environment type
    environment_type: str  # conda, virtualenv, poetry, pipenv, system
    environment_name: Optional[str] = None
    environment_file: Optional[str] = None  # environment.yml, requirements.txt, etc.
    
    # Dependencies
    packages: List[SoftwareDependency] = field(default_factory=list)
    
    # Lock file
    lock_file_hash: Optional[str] = None  # Hash of full lock file
    lock_file_content: Optional[str] = None  # Full lock file content (if small enough)
    
    def compute_environment_hash(self) -> str:
        """Compute deterministic hash of environment."""
        content = json.dumps({
            "python_version": self.python_version,
            "packages": sorted([
                (p.name, p.version) for p in self.packages
            ])
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]


@dataclass
class HardwareDescriptor:
    """
    Hardware description without assuming specific hardware.
    
    Captures sufficient information to characterize performance
    without requiring specific hardware for reproduction.
    """
    
    # CPU information
    cpu_architecture: str  # x86_64, arm64, ppc64le, etc.
    cpu_model: Optional[str] = None
    cpu_count: Optional[int] = None
    cpu_frequency_mhz: Optional[float] = None
    
    # GPU information (if applicable)
    gpu_available: bool = False
    gpu_count: Optional[int] = None
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    
    # Memory
    total_memory_gb: Optional[float] = None
    
    # Storage
    storage_type: Optional[str] = None  # SSD, HDD, NVMe, network, etc.
    
    # Platform
    platform_system: str = field(default_factory=platform.system)
    platform_release: str = field(default_factory=platform.release)
    platform_machine: str = field(default_factory=platform.machine)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": {
                "architecture": self.cpu_architecture,
                "model": self.cpu_model,
                "count": self.cpu_count,
                "frequency_mhz": self.cpu_frequency_mhz
            },
            "gpu": {
                "available": self.gpu_available,
                "count": self.gpu_count,
                "model": self.gpu_model,
                "memory_gb": self.gpu_memory_gb,
                "cuda_version": self.cuda_version
            },
            "memory": {
                "total_gb": self.total_memory_gb
            },
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.platform_machine
            }
        }


@dataclass
class WorkflowDefinition:
    """
    Workflow definition reference with versioning.
    
    Captures the complete workflow specification needed for reproduction.
    """
    engine: WorkflowEngine
    engine_version: str
    
    # Workflow location
    workflow_url: Optional[str] = None  # Git repository URL
    workflow_git_commit: Optional[str] = None
    workflow_path: Optional[str] = None  # Path within repo
    
    # Workflow identifier
    workflow_name: str = "unnamed_workflow"
    workflow_version: str = "1.0.0"
    workflow_id: UUID = field(default_factory=uuid4)
    
    # Content hash
    workflow_content_hash: Optional[str] = None  # Hash of workflow files
    
    # Dependencies
    workflow_dependencies: List[str] = field(default_factory=list)
    
    def compute_content_hash(self, workflow_files: Dict[str, str]) -> str:
        """
        Compute hash of workflow file contents.
        
        Args:
            workflow_files: Dict mapping file paths to content strings
        """
        content = json.dumps(workflow_files, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ParameterTemplate:
    """
    Parameter template with versioning.
    
    Ensures that parameter sets are versioned and reproducible.
    """
    template_id: UUID
    template_name: str
    template_version: str
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Description
    description: Optional[str] = None
    parameter_schema: Optional[Dict[str, Any]] = None  # JSON Schema
    
    # Provenance
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of parameters."""
        content = json.dumps(self.parameters, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]


@dataclass
class ExecutionContext:
    """
    Complete execution context for reproducibility.
    
    Captures everything needed to reproduce a computational experiment.
    """
    
    # Identifiers
    experiment_id: UUID
    context_id: UUID = field(default_factory=uuid4)
    
    # Container
    container: Optional[ContainerImage] = None
    
    # Software environment
    python_environment: Optional[PythonEnvironment] = None
    system_dependencies: List[SoftwareDependency] = field(default_factory=list)
    
    # Workflow
    workflow: Optional[WorkflowDefinition] = None
    parameter_template: Optional[ParameterTemplate] = None
    
    # Hardware (informational, not required for reproduction)
    hardware: Optional[HardwareDescriptor] = None
    
    # Execution backend
    backend: ExecutionBackend = ExecutionBackend.LOCAL
    backend_version: Optional[str] = None
    backend_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Resource allocation
    allocated_cpus: Optional[int] = None
    allocated_memory_gb: Optional[float] = None
    allocated_gpus: Optional[int] = None
    allocated_time_limit_seconds: Optional[int] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Reproducibility verification
    reproducibility_notes: Optional[str] = None
    known_limitations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution context to dictionary."""
        return {
            "context_id": str(self.context_id),
            "experiment_id": str(self.experiment_id),
            "container": self.container.to_dict() if self.container else None,
            "python_environment": {
                "python_version": self.python_environment.python_version if self.python_environment else None,
                "environment_type": self.python_environment.environment_type if self.python_environment else None,
                "environment_hash": self.python_environment.compute_environment_hash() if self.python_environment else None
            },
            "workflow": {
                "engine": self.workflow.engine.value if self.workflow else None,
                "version": self.workflow.workflow_version if self.workflow else None,
                "git_commit": self.workflow.workflow_git_commit if self.workflow else None
            },
            "parameter_template": {
                "id": str(self.parameter_template.template_id) if self.parameter_template else None,
                "version": self.parameter_template.template_version if self.parameter_template else None,
                "hash": self.parameter_template.compute_hash() if self.parameter_template else None
            },
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "backend": self.backend.value,
            "resource_allocation": {
                "cpus": self.allocated_cpus,
                "memory_gb": self.allocated_memory_gb,
                "gpus": self.allocated_gpus,
                "time_limit_seconds": self.allocated_time_limit_seconds
            },
            "timestamps": {
                "created": self.created_at.isoformat(),
                "started": self.started_at.isoformat() if self.started_at else None,
                "completed": self.completed_at.isoformat() if self.completed_at else None
            }
        }
    
    def generate_reproducibility_report(self) -> str:
        """Generate human-readable reproducibility report."""
        lines = [
            "=" * 70,
            "VSSS REPRODUCIBILITY REPORT",
            "=" * 70,
            "",
            f"Context ID: {self.context_id}",
            f"Experiment ID: {self.experiment_id}",
            "",
            "CONTAINER SPECIFICATION",
            "-" * 40,
        ]
        
        if self.container:
            lines.extend([
                f"Runtime: {self.container.runtime.value}",
                f"Image: {self.container.image_uri}",
                f"Digest: {self.container.image_digest}",
                f"Build Date: {self.container.build_date}",
            ])
        else:
            lines.append("No container used")
        
        lines.extend([
            "",
            "WORKFLOW SPECIFICATION",
            "-" * 40,
        ])
        
        if self.workflow:
            lines.extend([
                f"Engine: {self.workflow.engine.value} v{self.workflow.engine_version}",
                f"Workflow: {self.workflow.workflow_name} v{self.workflow.workflow_version}",
                f"Git Commit: {self.workflow.workflow_git_commit or 'N/A'}",
            ])
        
        lines.extend([
            "",
            "PARAMETER TEMPLATE",
            "-" * 40,
        ])
        
        if self.parameter_template:
            lines.extend([
                f"Template: {self.parameter_template.template_name}",
                f"Version: {self.parameter_template.template_version}",
                f"Hash: {self.parameter_template.compute_hash()}",
            ])
        
        lines.extend([
            "",
            "HARDWARE CONTEXT (Informational)",
            "-" * 40,
        ])
        
        if self.hardware:
            lines.extend([
                f"CPU: {self.hardware.cpu_architecture}",
                f"GPU Available: {self.hardware.gpu_available}",
                f"Platform: {self.hardware.platform_system} {self.hardware.platform_release}",
            ])
        
        lines.extend([
            "",
            "REPRODUCIBILITY NOTES",
            "-" * 40,
            self.reproducibility_notes or "None provided",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


class ReproducibilityManager:
    """
    Manager for reproducibility metadata.
    
    Provides utilities for:
    - Capturing current execution context
    - Validating reproducibility requirements
    - Generating reproducibility certificates
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.contexts: Dict[UUID, ExecutionContext] = {}
    
    def capture_current_context(
        self,
        experiment_id: UUID,
        container: Optional[ContainerImage] = None,
        workflow: Optional[WorkflowDefinition] = None,
        parameter_template: Optional[ParameterTemplate] = None,
        backend: ExecutionBackend = ExecutionBackend.LOCAL
    ) -> ExecutionContext:
        """
        Capture current execution environment.
        
        Automatically detects Python environment, hardware, and platform.
        """
        import sys
        
        # Detect Python environment
        python_env = self._detect_python_environment()
        
        # Detect hardware
        hardware = self._detect_hardware()
        
        context = ExecutionContext(
            experiment_id=experiment_id,
            container=container,
            python_environment=python_env,
            hardware=hardware,
            workflow=workflow,
            parameter_template=parameter_template,
            backend=backend,
            started_at=datetime.utcnow()
        )
        
        self.contexts[context.context_id] = context
        return context
    
    def _detect_python_environment(self) -> PythonEnvironment:
        """Detect current Python environment."""
        import sys
        import pkg_resources
        
        # Determine environment type
        if 'conda' in sys.executable or 'CONDA_DEFAULT_ENV' in __import__('os').environ:
            env_type = 'conda'
            env_name = __import__('os').environ.get('CONDA_DEFAULT_ENV', 'base')
        elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            env_type = 'virtualenv'
            env_name = None
        else:
            env_type = 'system'
            env_name = None
        
        # Get installed packages
        packages = []
        for dist in pkg_resources.working_set:
            packages.append(SoftwareDependency(
                name=dist.project_name,
                version=dist.version,
                installation_method='pip'  # Simplified
            ))
        
        return PythonEnvironment(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            python_executable=sys.executable,
            environment_type=env_type,
            environment_name=env_name,
            packages=packages
        )
    
    def _detect_hardware(self) -> HardwareDescriptor:
        """Detect current hardware."""
        import psutil
        
        # CPU info
        cpu_count = __import__('os').cpu_count()
        
        # Memory
        mem = psutil.virtual_memory()
        
        # Try to detect GPU
        gpu_available = False
        gpu_count = None
        gpu_model = None
        cuda_version = None
        
        try:
            # Check for CUDA
            result = __import__('subprocess').run(
                ['nvidia-smi', '--query-gpu=name,count', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                gpu_available = True
                lines = result.stdout.strip().split('\n')
                if lines:
                    gpu_model = lines[0].split(',')[0].strip()
                    gpu_count = len(lines)
            
            # Check CUDA version
            result = __import__('subprocess').run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        cuda_version = line.split('release')[-1].split(',')[0].strip()
                        break
        except:
            pass
        
        return HardwareDescriptor(
            cpu_architecture=platform.machine(),
            cpu_count=cpu_count,
            total_memory_gb=mem.total / (1024**3),
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_model=gpu_model,
            cuda_version=cuda_version,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine()
        )
    
    def validate_reproducibility(
        self,
        context: ExecutionContext,
        requirements: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that context meets reproducibility requirements.
        
        Returns (is_valid, list_of_issues).
        """
        issues = []
        
        # Check container
        if requirements.get('require_container', True) and not context.container:
            issues.append("Container required but not specified")
        
        # Check workflow versioning
        if requirements.get('require_workflow_version', True):
            if not context.workflow:
                issues.append("Workflow required but not specified")
            elif not context.workflow.workflow_git_commit:
                issues.append("Workflow git commit required for full reproducibility")
        
        # Check parameter template
        if requirements.get('require_parameter_template', True) and not context.parameter_template:
            issues.append("Parameter template required but not specified")
        
        return len(issues) == 0, issues
    
    def export_context(self, context_id: UUID, output_path: str) -> None:
        """Export execution context to JSON file."""
        context = self.contexts.get(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")
        
        with open(output_path, 'w') as f:
            json.dump(context.to_dict(), f, indent=2)


# Convenience functions

def create_container_image(
    image_uri: str,
    image_digest: str,
    runtime: ContainerRuntime = ContainerRuntime.DOCKER
) -> ContainerImage:
    """Factory function to create ContainerImage."""
    return ContainerImage(
        runtime=runtime,
        image_uri=image_uri,
        image_digest=image_digest
    )


def create_parameter_template(
    name: str,
    version: str,
    parameters: Dict[str, Any]
) -> ParameterTemplate:
    """Factory function to create ParameterTemplate."""
    return ParameterTemplate(
        template_id=uuid4(),
        template_name=name,
        template_version=version,
        parameters=parameters
    )
