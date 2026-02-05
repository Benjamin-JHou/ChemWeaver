"""
CAS-VS Workflow Orchestration Integration

Provides compatibility with major workflow engines:
- Nextflow
- Snakemake
- Common Workflow Language (CWL)

Features:
- Container-native execution
- Checkpoint recovery
- Stage-level caching
- Cross-engine compatibility

Author: CAS-VS Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class WorkflowEngine(Enum):
    """Supported workflow engines."""
    NEXTFLOW = "nextflow"
    SNAKEMAKE = "snakemake"
    CWL = "cwl"
    PREFECT = "prefect"
    AIRFLOW = "airflow"


@dataclass
class ContainerSpec:
    """Container specification for workflow tasks."""
    image: str
    registry: Optional[str] = None
    tag: str = "latest"
    pull_policy: str = "if_not_present"  # always, never, if_not_present
    
    # Resource limits within container
    cpu_limit: Optional[float] = None
    memory_limit_gb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "tag": self.tag,
            "registry": self.registry,
            "pull_policy": self.pull_policy
        }


@dataclass
class WorkflowTask:
    """Single task in a workflow."""
    task_id: UUID = field(default_factory=uuid4)
    task_name: str = "unnamed_task"
    
    # Execution
    command: str = ""
    arguments: List[str] = field(default_factory=list)
    working_dir: str = "./"
    
    # Container
    container: Optional[ContainerSpec] = None
    
    # Resources
    cpu_cores: int = 1
    memory_gb: float = 4.0
    gpu_count: int = 0
    time_limit_hours: Optional[float] = None
    
    # I/O
    input_files: Dict[str, str] = field(default_factory=dict)
    output_files: Dict[str, str] = field(default_factory=dict)
    
    # Dependencies
    depends_on: List[UUID] = field(default_factory=list)
    
    # Caching
    cache_key: Optional[str] = None
    cache_enabled: bool = True
    
    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_interval_minutes: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "task_name": self.task_name,
            "command": self.command,
            "resources": {
                "cpu": self.cpu_cores,
                "memory_gb": self.memory_gb,
                "gpu": self.gpu_count
            },
            "dependencies": [str(d) for d in self.depends_on]
        }


class WorkflowGenerator(ABC):
    """Abstract base class for workflow generators."""
    
    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
        self.workflow_id = uuid4()
    
    @abstractmethod
    def generate_workflow(
        self,
        tasks: List[WorkflowTask],
        output_path: str
    ) -> str:
        """Generate workflow file for the specific engine."""
        pass
    
    @abstractmethod
    def generate_execution_command(
        self,
        workflow_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate command to execute the workflow."""
        pass


class NextflowGenerator(WorkflowGenerator):
    """Generate Nextflow workflows."""
    
    def __init__(self):
        super().__init__(WorkflowEngine.NEXTFLOW)
    
    def generate_workflow(
        self,
        tasks: List[WorkflowTask],
        output_path: str
    ) -> str:
        """Generate Nextflow DSL2 workflow."""
        lines = [
            "#!/usr/bin/env nextflow",
            "",
            "// CAS-VS Generated Nextflow Workflow",
            f"// Workflow ID: {self.workflow_id}",
            "",
            "nextflow.enable.dsl=2",
            "",
            "// Parameters",
            "params.input_library = 'library.sdf'",
            "params.target_pdb = 'target.pdb'",
            "params.output_dir = 'results'",
            "",
            "// Processes"
        ]
        
        # Generate processes for each task
        for task in tasks:
            lines.extend(self._generate_process(task))
            lines.append("")
        
        # Generate workflow
        lines.extend([
            "// Workflow Definition",
            "workflow {",
            "    // Input channel",
            "    library_ch = Channel.fromPath(params.input_library)",
            "",
            "    // Execute stages"
        ])
        
        prev_result = "library_ch"
        for i, task in enumerate(tasks):
            task_var = f"stage_{i}_result"
            lines.append(f"    {task_var} = {task.task_name}({prev_result})")
            prev_result = task_var
        
        lines.extend([
            "}",
            ""
        ])
        
        # Write to file
        content = "\n".join(lines)
        Path(output_path).write_text(content)
        
        return output_path
    
    def _generate_process(self, task: WorkflowTask) -> List[str]:
        """Generate Nextflow process definition."""
        lines = [
            f"process {task.task_name} {{"
        ]
        
        # Container directive
        if task.container:
            lines.append(f"    container '{task.container.image}:{task.container.tag}'")
        
        # Resource directives
        lines.append("")
        lines.append("    // Resource configuration")
        lines.append(f"    cpus {task.cpu_cores}")
        lines.append(f"    memory '{task.memory_gb} GB'")
        
        if task.gpu_count > 0:
            lines.append(f"    accelerator 1, type: 'gpu'")
        
        if task.time_limit_hours:
            lines.append(f"    time '{task.time_limit_hours}h'")
        
        # Cache directive
        if task.cache_enabled and task.cache_key:
            lines.append(f"    cache 'lenient'")
        
        # Input/Output
        lines.extend([
            "",
            "    input:",
            "    path input_file",
            "",
            "    output:",
            "    path 'output.*'"
        ])
        
        # Script
        lines.extend([
            "",
            "    script:",
            "    \"\"\"",
            f"    {task.command} {' '.join(task.arguments)}",
            "    \"\"\""
        ])
        
        lines.append("}")
        
        return lines
    
    def generate_execution_command(
        self,
        workflow_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate Nextflow execution command."""
        cmd_parts = ["nextflow run", workflow_path]
        
        if config:
            for key, value in config.items():
                cmd_parts.append(f"--{key} {value}")
        
        cmd_parts.append("-with-report report.html")
        cmd_parts.append("-with-trace trace.txt")
        
        return " ".join(cmd_parts)


class SnakemakeGenerator(WorkflowGenerator):
    """Generate Snakemake workflows."""
    
    def __init__(self):
        super().__init__(WorkflowEngine.SNAKEMAKE)
    
    def generate_workflow(
        self,
        tasks: List[WorkflowTask],
        output_path: str
    ) -> str:
        """Generate Snakemake workflow file."""
        lines = [
            "# CAS-VS Generated Snakemake Workflow",
            f"# Workflow ID: {self.workflow_id}",
            "",
            "import os",
            "",
            "# Configuration",
            'configfile: "config.yaml"',
            "",
            "# Wildcard constraints",
            'wildcard_constraints:',
            '    stage="[0-9]+",',
            ""
        ]
        
        # Generate rules
        for i, task in enumerate(tasks):
            lines.extend(self._generate_rule(task, i))
            lines.append("")
        
        # Write to file
        content = "\n".join(lines)
        Path(output_path).write_text(content)
        
        # Generate config file
        config_path = Path(output_path).parent / "config.yaml"
        config_content = {
            "input_library": "library.sdf",
            "target_pdb": "target.pdb",
            "output_dir": "results",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16
            }
        }
        config_path.write_text(json.dumps(config_content, indent=2))
        
        return output_path
    
    def _generate_rule(self, task: WorkflowTask, index: int) -> List[str]:
        """Generate Snakemake rule definition."""
        rule_name = task.task_name.lower().replace(" ", "_")
        
        lines = [
            f"rule {rule_name}:"
        ]
        
        # Input
        if index == 0:
            lines.append('    input:')
            lines.append('        config["input_library"]')
        else:
            lines.append(f'    input:')
            lines.append(f'        "results/stage_{index-1}/output.txt"')
        
        # Output
        lines.append(f'    output:')
        lines.append(f'        "results/stage_{index}/output.txt"')
        
        # Log
        lines.append(f'    log:')
        lines.append(f'        "logs/stage_{index}.log"')
        
        # Resources
        lines.append(f'    resources:')
        lines.append(f'        cpus={task.cpu_cores},')
        lines.append(f'        mem_mb={int(task.memory_gb * 1024)}')
        
        if task.gpu_count > 0:
            lines.append(f'    resources:')
            lines.append(f'        gpus={task.gpu_count}')
        
        # Container
        if task.container:
            lines.append(f'    container:')
            lines.append(f'        "docker://{task.container.image}:{task.container.tag}"')
        
        # Shell command
        lines.append(f'    shell:')
        cmd = f"{task.command} {' '.join(task.arguments)}"
        lines.append(f'        "{cmd} > {{output}} 2> {{log}}"')
        
        return lines
    
    def generate_execution_command(
        self,
        workflow_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate Snakemake execution command."""
        cmd_parts = ["snakemake"]
        
        if config:
            cores = config.get("cores", 4)
            cmd_parts.append(f"--cores {cores}")
        else:
            cmd_parts.append("--cores all")
        
        cmd_parts.append("--use-conda")
        cmd_parts.append("--rerun-incomplete")
        cmd_parts.append("--summary")
        
        return " ".join(cmd_parts)


class CWLGenerator(WorkflowGenerator):
    """Generate CWL (Common Workflow Language) workflows."""
    
    def __init__(self):
        super().__init__(WorkflowEngine.CWL)
    
    def generate_workflow(
        self,
        tasks: List[WorkflowTask],
        output_path: str
    ) -> str:
        """Generate CWL workflow."""
        workflow = {
            "cwlVersion": "v1.2",
            "class": "Workflow",
            "id": str(self.workflow_id),
            "label": "CAS-VS Multi-Stage Screening",
            "doc": "Compute-Adaptive Virtual Screening Pipeline",
            "inputs": {
                "input_library": {
                    "type": "File",
                    "doc": "Input compound library"
                },
                "target_structure": {
                    "type": "File",
                    "doc": "Target protein structure"
                }
            },
            "outputs": {
                "final_results": {
                    "type": "File",
                    "outputSource": f"stage_{len(tasks)-1}/output"
                }
            },
            "steps": {}
        }
        
        # Generate steps
        for i, task in enumerate(tasks):
            step_name = f"stage_{i}"
            
            workflow["steps"][step_name] = {
                "run": f"tools/{task.task_name}.cwl",
                "in": {
                    "input": f"stage_{i-1}/output" if i > 0 else "input_library"
                },
                "out": ["output"]
            }
        
        # Write to file
        content = json.dumps(workflow, indent=2)
        Path(output_path).write_text(content)
        
        # Generate tool definitions
        tools_dir = Path(output_path).parent / "tools"
        tools_dir.mkdir(exist_ok=True)
        
        for task in tasks:
            tool_def = self._generate_tool_definition(task)
            tool_path = tools_dir / f"{task.task_name}.cwl"
            tool_path.write_text(json.dumps(tool_def, indent=2))
        
        return output_path
    
    def _generate_tool_definition(self, task: WorkflowTask) -> Dict[str, Any]:
        """Generate CWL tool definition."""
        tool = {
            "cwlVersion": "v1.2",
            "class": "CommandLineTool",
            "baseCommand": task.command.split(),
            "inputs": {
                "input": {
                    "type": "File",
                    "inputBinding": {"position": 1}
                }
            },
            "outputs": {
                "output": {
                    "type": "File",
                    "outputBinding": {"glob": "output.*"}
                }
            },
            "requirements": {
                "ResourceRequirement": {
                    "coresMin": task.cpu_cores,
                    "ramMin": int(task.memory_gb * 1024)
                }
            }
        }
        
        if task.container:
            tool["hints"] = {
                "DockerRequirement": {
                    "dockerPull": f"{task.container.image}:{task.container.tag}"
                }
            }
        
        return tool
    
    def generate_execution_command(
        self,
        workflow_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate CWL execution command."""
        cmd_parts = ["cwltool"]
        
        if config and config.get("parallel"):
            cmd_parts.append("--parallel")
        
        cmd_parts.append(workflow_path)
        
        # Add inputs
        cmd_parts.append("--input_library library.sdf")
        cmd_parts.append("--target_structure target.pdb")
        
        return " ".join(cmd_parts)


class WorkflowOrchestrator:
    """
    Workflow Orchestration Integration (MANDATORY COMPONENT)
    
    Provides unified interface to generate and execute workflows
    across different orchestration engines.
    """
    
    def __init__(self):
        self.orchestrator_id = uuid4()
        self.generators: Dict[WorkflowEngine, WorkflowGenerator] = {
            WorkflowEngine.NEXTFLOW: NextflowGenerator(),
            WorkflowEngine.SNAKEMAKE: SnakemakeGenerator(),
            WorkflowEngine.CWL: CWLGenerator()
        }
        self.active_engine: Optional[WorkflowEngine] = None
    
    def set_engine(self, engine: WorkflowEngine) -> None:
        """Set active workflow engine."""
        if engine not in self.generators:
            raise ValueError(f"Unsupported engine: {engine}")
        self.active_engine = engine
    
    def generate_workflow(
        self,
        tasks: List[WorkflowTask],
        output_path: str,
        engine: Optional[WorkflowEngine] = None
    ) -> str:
        """Generate workflow for specified engine."""
        target_engine = engine or self.active_engine
        if not target_engine:
            raise ValueError("No workflow engine specified")
        
        generator = self.generators[target_engine]
        return generator.generate_workflow(tasks, output_path)
    
    def get_execution_command(
        self,
        workflow_path: str,
        config: Optional[Dict[str, Any]] = None,
        engine: Optional[WorkflowEngine] = None
    ) -> str:
        """Get execution command for workflow."""
        target_engine = engine or self.active_engine
        if not target_engine:
            raise ValueError("No workflow engine specified")
        
        generator = self.generators[target_engine]
        return generator.generate_execution_command(workflow_path, config)
    
    @staticmethod
    def create_casvs_pipeline_tasks(
        stages_config: List[Dict[str, Any]]
    ) -> List[WorkflowTask]:
        """
        Create workflow tasks for CAS-VS 5-stage pipeline.
        
        Args:
            stages_config: List of stage configurations
            
        Returns:
            List of workflow tasks
        """
        tasks = []
        
        stage_tools = {
            0: ("standardization", "vsss/standardization:latest"),
            1: ("ligand_filtering", "vsss/filtering:latest"),
            2: ("pharmacophore", "vsss/pharmacophore:latest"),
            3: ("docking", "vsss/docking:latest"),
            4: ("ai_rescoring", "vsss/ai:latest")
        }
        
        for i, config in enumerate(stages_config):
            tool_name, image = stage_tools.get(i, (f"stage_{i}", "ubuntu:20.04"))
            
            task = WorkflowTask(
                task_name=tool_name,
                command=config.get("command", "casvs-stage"),
                arguments=config.get("args", []),
                container=ContainerSpec(image=image),
                cpu_cores=config.get("cpu", 1),
                memory_gb=config.get("memory", 4.0),
                gpu_count=config.get("gpu", 0),
                cache_enabled=config.get("cache", True)
            )
            
            tasks.append(task)
        
        return tasks


# Export
__all__ = [
    'WorkflowEngine',
    'ContainerSpec',
    'WorkflowTask',
    'WorkflowGenerator',
    'NextflowGenerator',
    'SnakemakeGenerator',
    'CWLGenerator',
    'WorkflowOrchestrator'
]
