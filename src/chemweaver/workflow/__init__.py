"""
ChemWeaver Workflow Module

This module provides the compute-adaptive screening workflow system for ChemWeaver,
including the workflow engine, orchestration, execution, and workflow stages.

Author: ChemWeaver Development Team
Version: 1.0.0
"""

# Smart dependency loading
from ..utils.dependencies import deps

# Check workflow dependencies
workflow_deps = deps.get_data_status()
has_full_workflow = all(workflow_deps.values())

if has_full_workflow:
    # Full workflow functionality available
    try:
        from .cas_engine import *
        from .orchestration import *
        from .execution import *
        from .stages import *
        
        _FULL_WORKFLOW_AVAILABLE = True
        print("✅ ChemWeaver-Workflow: Full functionality available")
        
    except Exception as e:
        print(f"⚠️ ChemWeaver-Workflow: Import error - {e}")
        _FULL_WORKFLOW_AVAILABLE = False
else:
    # Missing dependencies - provide fallback
    print("⚠️ ChemWeaver-Workflow: Limited functionality due to missing dependencies")
    missing = [k for k, v in workflow_deps.items() if not v]
    print(f"   Missing: {', '.join(missing)}")
    print("   Install with: pip install h5py pyarrow")
    _FULL_WORKFLOW_AVAILABLE = False

# Always provide basic workflow functionality
from ..core.pipeline import MinimalScreeningPipeline

__all__ = [
    # Always available
    "MinimalScreeningPipeline",
]

# Full workflow exports (when available)
if _FULL_WORKFLOW_AVAILABLE:
    __all__.extend([
        # Core workflow components
        "ChemWeaverEngine",
        "WorkflowOrchestrator", 
        "ExecutionManager",
        "WorkflowStages",
        
        # Strategy optimization
        "ResourceDescriptor",
        "TargetDifficulty",
        "ResourceClass",
        
        # Execution backends
        "ComputeBackend",
        "JobScheduler",
    ])

# Status flag
_full_workflow_available = _FULL_WORKFLOW_AVAILABLE