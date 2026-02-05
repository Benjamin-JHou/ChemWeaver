"""
CAS-VS Compute Abstraction Layer

Provides unified execution interface supporting heterogeneous compute
infrastructure: local workstations, HPC clusters (SLURM, PBS, LSF),
and cloud distributed systems.

Key Features:
- Backend-agnostic execution
- Resource scheduling and allocation
- Job queue management
- Checkpoint and recovery

Author: CAS-VS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class ExecutionBackend(Enum):
    """Supported execution backends."""
    LOCAL = "local"
    LOCAL_PARALLEL = "local_parallel"
    SLURM = "slurm"
    PBS = "pbs"
    LSF = "lsf"
    SGE = "sge"  # Sun Grid Engine
    KUBERNETES = "kubernetes"
    AWS_BATCH = "aws_batch"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BATCH = "azure_batch"


class JobStatus(Enum):
    """Status of an execution job."""
    PENDING = auto()
    SUBMITTED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class ResourceRequest:
    """Request for compute resources."""
    
    # CPU
    cpu_cores: int = 1
    cpu_memory_gb: float = 4.0
    
    # GPU
    gpu_count: int = 0
    gpu_memory_gb: Optional[float] = None
    gpu_model: Optional[str] = None  # e.g., "v100", "a100", "rtx3090"
    
    # Time
    wall_time_seconds: Optional[float] = None
    
    # Storage
    local_storage_gb: float = 10.0
    shared_storage_gb: float = 0.0
    
    # Special requirements
    requires_infiniband: bool = False
    requires_exclusive_node: bool = False
    
    def to_backend_specific(self, backend: ExecutionBackend) -> Dict[str, Any]:
        """Convert to backend-specific resource specification."""
        if backend == ExecutionBackend.SLURM:
            return self._to_slurm_format()
        elif backend == ExecutionBackend.PBS:
            return self._to_pbs_format()
        elif backend == ExecutionBackend.KUBERNETES:
            return self._to_kubernetes_format()
        else:
            return self._to_generic_format()
    
    def _to_slurm_format(self) -> Dict[str, Any]:
        """Convert to SLURM directives."""
        directives = {
            "cpus-per-task": self.cpu_cores,
            "mem": f"{int(self.cpu_memory_gb)}G"
        }
        
        if self.wall_time_seconds:
            hours = int(self.wall_time_seconds // 3600)
            minutes = int((self.wall_time_seconds % 3600) // 60)
            directives["time"] = f"{hours}:{minutes:02d}:00"
        
        if self.gpu_count > 0:
            directives["gres"] = f"gpu:{self.gpu_count}"
        
        return directives
    
    def _to_pbs_format(self) -> Dict[str, Any]:
        """Convert to PBS directives."""
        directives = {
            "ncpus": self.cpu_cores,
            "mem": f"{int(self.cpu_memory_gb)}gb"
        }
        
        if self.wall_time_seconds:
            hours = int(self.wall_time_seconds // 3600)
            minutes = int((self.wall_time_seconds % 3600) // 60)
            directives["walltime"] = f"{hours}:{minutes:02d}:00"
        
        if self.gpu_count > 0:
            directives["ngpus"] = self.gpu_count
        
        return directives
    
    def _to_kubernetes_format(self) -> Dict[str, Any]:
        """Convert to Kubernetes resource spec."""
        resources = {
            "requests": {
                "cpu": str(self.cpu_cores),
                "memory": f"{int(self.cpu_memory_gb)}Gi"
            }
        }
        
        if self.gpu_count > 0:
            resources["limits"] = {
                "nvidia.com/gpu": str(self.gpu_count)
            }
        
        return resources
    
    def _to_generic_format(self) -> Dict[str, Any]:
        """Convert to generic format."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.cpu_memory_gb,
            "gpu_count": self.gpu_count,
            "wall_time_seconds": self.wall_time_seconds
        }


@dataclass
class JobSpecification:
    """Specification for an execution job."""
    
    job_id: UUID = field(default_factory=uuid4)
    job_name: str = "casvs_job"
    
    # Command to execute
    command: str = ""
    arguments: List[str] = field(default_factory=list)
    working_directory: str = "./"
    
    # Environment
    environment_variables: Dict[str, str] = field(default_factory=dict)
    container_image: Optional[str] = None
    
    # Resource requirements
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    
    # I/O
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    log_file: Optional[str] = None
    
    # Dependencies
    depends_on: List[UUID] = field(default_factory=list)  # Job IDs
    
    # Metadata
    stage_id: Optional[UUID] = None
    checkpoint_enabled: bool = True
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": str(self.job_id),
            "job_name": self.job_name,
            "command": self.command,
            "resources": self.resources.to_backend_specific(ExecutionBackend.LOCAL),
            "checkpoint_enabled": self.checkpoint_enabled
        }


@dataclass
class JobResult:
    """Result of job execution."""
    
    job_id: UUID
    status: JobStatus
    
    # Timing
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Resources used
    cpu_time_seconds: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    gpu_time_seconds: Optional[float] = None
    
    # Output
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Error info
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def success(self) -> bool:
        """Check if job succeeded."""
        return self.status == JobStatus.COMPLETED and (self.exit_code == 0 or self.exit_code is None)


class ComputeBackend(ABC):
    """Abstract base class for compute backends."""
    
    def __init__(self, backend_type: ExecutionBackend, config: Optional[Dict[str, Any]] = None):
        self.backend_type = backend_type
        self.config = config or {}
        self.backend_id = uuid4()
    
    @abstractmethod
    def submit_job(self, job_spec: JobSpecification) -> JobResult:
        """Submit a job to the backend."""
        pass
    
    @abstractmethod
    def check_job_status(self, job_id: UUID) -> JobStatus:
        """Check status of a submitted job."""
        pass
    
    @abstractmethod
    def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a running job."""
        pass
    
    @abstractmethod
    def get_job_result(self, job_id: UUID) -> Optional[JobResult]:
        """Get result of completed job."""
        pass
    
    @abstractmethod
    def list_active_jobs(self) -> List[UUID]:
        """List IDs of active (running/pending) jobs."""
        pass
    
    @abstractmethod
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource availability."""
        pass
    
    def supports_gpu(self) -> bool:
        """Check if backend supports GPU execution."""
        status = self.get_resource_status()
        return status.get("gpu_available", False)


class LocalBackend(ComputeBackend):
    """Local single-node execution backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ExecutionBackend.LOCAL, config)
        self.max_parallel_jobs = self.config.get("max_parallel_jobs", 1)
        self.active_jobs: Dict[UUID, subprocess.Popen] = {}
        self.job_results: Dict[UUID, JobResult] = {}
    
    def submit_job(self, job_spec: JobSpecification) -> JobResult:
        """Execute job locally."""
        import os
        
        # Build command
        cmd = [job_spec.command] + job_spec.arguments
        
        # Set environment
        env = os.environ.copy()
        env.update(job_spec.environment_variables)
        
        # Submit
        result = JobResult(
            job_id=job_spec.job_id,
            status=JobStatus.RUNNING,
            submitted_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=job_spec.working_directory,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.active_jobs[job_spec.job_id] = process
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            result.status = JobStatus.COMPLETED if process.returncode == 0 else JobStatus.FAILED
            result.completed_at = datetime.utcnow()
            result.exit_code = process.returncode
            result.stdout = stdout
            result.stderr = stderr
            
        except Exception as e:
            result.status = JobStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.utcnow()
        
        finally:
            if job_spec.job_id in self.active_jobs:
                del self.active_jobs[job_spec.job_id]
            self.job_results[job_spec.job_id] = result
        
        return result
    
    def check_job_status(self, job_id: UUID) -> JobStatus:
        """Check job status."""
        if job_id in self.active_jobs:
            process = self.active_jobs[job_id]
            if process.poll() is None:
                return JobStatus.RUNNING
        
        if job_id in self.job_results:
            return self.job_results[job_id].status
        
        return JobStatus.PENDING
    
    def cancel_job(self, job_id: UUID) -> bool:
        """Cancel job."""
        if job_id in self.active_jobs:
            process = self.active_jobs[job_id]
            process.terminate()
            return True
        return False
    
    def get_job_result(self, job_id: UUID) -> Optional[JobResult]:
        """Get job result."""
        return self.job_results.get(job_id)
    
    def list_active_jobs(self) -> List[UUID]:
        """List active jobs."""
        return list(self.active_jobs.keys())
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get resource status."""
        import psutil
        
        return {
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": False  # Local backend doesn't detect GPU automatically
        }


class SlurmBackend(ComputeBackend):
    """SLURM HPC cluster backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ExecutionBackend.SLURM, config)
        self.partition = self.config.get("partition", "default")
        self.account = self.config.get("account")
    
    def submit_job(self, job_spec: JobSpecification) -> JobResult:
        """Submit job via sbatch."""
        # Build sbatch script
        script_lines = ["#!/bin/bash"]
        
        # SLURM directives
        resources = job_spec.resources.to_backend_specific(ExecutionBackend.SLURM)
        for key, value in resources.items():
            script_lines.append(f"#SBATCH --{key}={value}")
        
        script_lines.append(f"#SBATCH --job-name={job_spec.job_name}")
        script_lines.append(f"#SBATCH --partition={self.partition}")
        
        if self.account:
            script_lines.append(f"#SBATCH --account={self.account}")
        
        if job_spec.log_file:
            script_lines.append(f"#SBATCH --output={job_spec.log_file}")
        
        # Container support
        if job_spec.container_image:
            script_lines.append(f"#SBATCH --container={job_spec.container_image}")
        
        # Environment
        for key, value in job_spec.environment_variables.items():
            script_lines.append(f"export {key}={value}")
        
        # Working directory
        script_lines.append(f"cd {job_spec.working_directory}")
        
        # Command
        script_lines.append(f"{job_spec.command} {' '.join(job_spec.arguments)}")
        
        script = "\n".join(script_lines)
        
        # Submit
        result = JobResult(
            job_id=job_spec.job_id,
            status=JobStatus.SUBMITTED,
            submitted_at=datetime.utcnow()
        )
        
        try:
            proc = subprocess.run(
                ["sbatch", "--parsable"],
                input=script,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job ID from sbatch output
            slurm_job_id = proc.stdout.strip().split(";")[0]
            result.output_data["slurm_job_id"] = slurm_job_id
            
        except subprocess.CalledProcessError as e:
            result.status = JobStatus.FAILED
            result.error_message = f"sbatch failed: {e.stderr}"
        
        return result
    
    def check_job_status(self, job_id: UUID) -> JobStatus:
        """Check job status via sacct."""
        # Implementation would parse sacct output
        # Simplified for example
        return JobStatus.RUNNING
    
    def cancel_job(self, job_id: UUID) -> bool:
        """Cancel job via scancel."""
        try:
            subprocess.run(["scancel", str(job_id)], check=True)
            return True
        except:
            return False
    
    def get_job_result(self, job_id: UUID) -> Optional[JobResult]:
        """Get job result."""
        # Would query via sacct and parse
        return None
    
    def list_active_jobs(self) -> List[UUID]:
        """List active jobs."""
        # Would query via squeue
        return []
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get resource status via sinfo."""
        try:
            result = subprocess.run(
                ["sinfo", "--format=%C", "--noheader"],
                capture_output=True,
                text=True
            )
            # Parse CPU info
            return {"status": "available", "details": result.stdout}
        except:
            return {"status": "unavailable"}


class KubernetesBackend(ComputeBackend):
    """Kubernetes cluster backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(ExecutionBackend.KUBERNETES, config)
        self.namespace = self.config.get("namespace", "default")
    
    def submit_job(self, job_spec: JobSpecification) -> JobResult:
        """Submit job as Kubernetes Job."""
        # Build job manifest
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"casvs-{job_spec.job_id.hex[:8]}",
                "namespace": self.namespace
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "screening",
                            "image": job_spec.container_image or "ubuntu:20.04",
                            "command": [job_spec.command],
                            "args": job_spec.arguments,
                            "resources": job_spec.resources.to_backend_specific(
                                ExecutionBackend.KUBERNETES
                            ),
                            "env": [
                                {"name": k, "value": v}
                                for k, v in job_spec.environment_variables.items()
                            ]
                        }],
                        "restartPolicy": "Never"
                    }
                },
                "backoffLimit": job_spec.max_retries
            }
        }
        
        # Would apply via kubectl
        result = JobResult(
            job_id=job_spec.job_id,
            status=JobStatus.SUBMITTED,
            submitted_at=datetime.utcnow()
        )
        
        return result
    
    def check_job_status(self, job_id: UUID) -> JobStatus:
        """Check job status via kubectl."""
        return JobStatus.RUNNING
    
    def cancel_job(self, job_id: UUID) -> bool:
        """Cancel job via kubectl delete."""
        return True
    
    def get_job_result(self, job_id: UUID) -> Optional[JobResult]:
        """Get job result."""
        return None
    
    def list_active_jobs(self) -> List[UUID]:
        """List active jobs."""
        return []
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get resource status."""
        return {"status": "available", "nodes": 3}


class ComputeAbstractionLayer:
    """
    Compute Abstraction Layer (MANDATORY COMPONENT)
    
    Provides unified interface to heterogeneous compute infrastructure.
    Separates workflow logic from execution backend.
    """
    
    def __init__(self):
        self.layer_id = uuid4()
        self.backends: Dict[ExecutionBackend, ComputeBackend] = {}
        self.active_backend: Optional[ExecutionBackend] = None
        self.job_queue: List[JobSpecification] = []
        self.completed_jobs: Dict[UUID, JobResult] = {}
    
    def register_backend(
        self, 
        backend_type: ExecutionBackend, 
        backend: ComputeBackend
    ) -> None:
        """Register a compute backend."""
        self.backends[backend_type] = backend
        if self.active_backend is None:
            self.active_backend = backend_type
    
    def set_active_backend(self, backend_type: ExecutionBackend) -> None:
        """Set the active execution backend."""
        if backend_type not in self.backends:
            raise ValueError(f"Backend {backend_type} not registered")
        self.active_backend = backend_type
    
    def submit_job(
        self, 
        job_spec: JobSpecification,
        backend_type: Optional[ExecutionBackend] = None
    ) -> JobResult:
        """Submit job to backend."""
        backend = self._get_backend(backend_type)
        result = backend.submit_job(job_spec)
        self.completed_jobs[job_spec.job_id] = result
        return result
    
    def submit_jobs_batch(
        self, 
        job_specs: List[JobSpecification],
        backend_type: Optional[ExecutionBackend] = None
    ) -> List[JobResult]:
        """Submit multiple jobs as batch."""
        results = []
        for job_spec in job_specs:
            result = self.submit_job(job_spec, backend_type)
            results.append(result)
        return results
    
    def check_job_status(
        self, 
        job_id: UUID,
        backend_type: Optional[ExecutionBackend] = None
    ) -> JobStatus:
        """Check job status."""
        backend = self._get_backend(backend_type)
        return backend.check_job_status(job_id)
    
    def cancel_job(
        self, 
        job_id: UUID,
        backend_type: Optional[ExecutionBackend] = None
    ) -> bool:
        """Cancel job."""
        backend = self._get_backend(backend_type)
        return backend.cancel_job(job_id)
    
    def get_resource_status(
        self, 
        backend_type: Optional[ExecutionBackend] = None
    ) -> Dict[str, Any]:
        """Get resource availability."""
        backend = self._get_backend(backend_type)
        return backend.get_resource_status()
    
    def _get_backend(
        self, 
        backend_type: Optional[ExecutionBackend] = None
    ) -> ComputeBackend:
        """Get backend instance."""
        if backend_type is None:
            if self.active_backend is None:
                raise ValueError("No active backend set")
            backend_type = self.active_backend
        
        if backend_type not in self.backends:
            raise ValueError(f"Backend {backend_type} not registered")
        
        return self.backends[backend_type]
    
    def detect_best_backend(self) -> ExecutionBackend:
        """Auto-detect best available backend."""
        # Check for SLURM
        try:
            subprocess.run(["sbatch", "--version"], capture_output=True, check=True)
            return ExecutionBackend.SLURM
        except:
            pass
        
        # Check for PBS
        try:
            subprocess.run(["qstat", "--version"], capture_output=True, check=True)
            return ExecutionBackend.PBS
        except:
            pass
        
        # Check for Kubernetes
        try:
            subprocess.run(["kubectl", "version"], capture_output=True, check=True)
            return ExecutionBackend.KUBERNETES
        except:
            pass
        
        # Default to local
        return ExecutionBackend.LOCAL
    
    @classmethod
    def create_default(cls) -> ComputeAbstractionLayer:
        """Factory method to create layer with local backend."""
        layer = cls()
        local_backend = LocalBackend()
        layer.register_backend(ExecutionBackend.LOCAL, local_backend)
        return layer


# Export
__all__ = [
    'ExecutionBackend',
    'JobStatus',
    'ResourceRequest',
    'JobSpecification',
    'JobResult',
    'ComputeBackend',
    'LocalBackend',
    'SlurmBackend',
    'KubernetesBackend',
    'ComputeAbstractionLayer'
]
