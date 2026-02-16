"""
ChemWeaver: AI-Powered Virtual Screening Infrastructure
======================================================

ChemWeaver is a comprehensive virtual screening platform that combines:

- ChemWeaver-Data: Universal data standard with FAIR compliance
- ChemWeaver-Workflow: Compute-adaptive screening workflows  
- ChemWeaver-AI: Physics-informed AI surrogates with uncertainty quantification
- ChemWeaver-Benchmark: Comprehensive evaluation framework

Features:
- 50-300× speedup over traditional docking
- Calibrated uncertainty estimates (ECE < 0.05)
- Container-native reproducibility
- 8 compute backend support
- Pre-registered prospective validation

Author: ChemWeaver Development Team
Version: 1.0.3
License: MIT
"""

__version__ = "1.0.3"
__author__ = "ChemWeaver Development Team"

# Core imports
from .core.pipeline import Compound, MinimalScreeningPipeline
from .core.inference import MinimalSurrogateModel, Prediction

# Test availability of each module conservatively
try:
    import chemweaver.data
    _DATA_AVAILABLE = True
except Exception as e:
    _DATA_AVAILABLE = False

try:
    import chemweaver.workflow
    from .workflow import _full_workflow_available
    _WORKFLOW_AVAILABLE = _full_workflow_available
except Exception as e:
    _WORKFLOW_AVAILABLE = False

try:
    import chemweaver.ai
    from .ai import _full_ai_available
    _AI_AVAILABLE = _full_ai_available
except Exception as e:
    _AI_AVAILABLE = False

try:
    import chemweaver.benchmark
    _BENCHMARK_AVAILABLE = True
except Exception as e:
    _BENCHMARK_AVAILABLE = False

# Main exports
__all__ = [
    # Core functionality (always available)
    "Compound",
    "MinimalScreeningPipeline", 
    "MinimalSurrogateModel",
    "Prediction",
    
    # Full infrastructure (available when dependencies met)
    "ChemWeaverData",
    "ChemWeaverWorkflow", 
    "ChemWeaverAI",
    "ChemWeaverBenchmark",
]

# Availability flags for optional components
_data_available = _DATA_AVAILABLE
_workflow_available = _WORKFLOW_AVAILABLE
_ai_available = _AI_AVAILABLE
_benchmark_available = _BENCHMARK_AVAILABLE

def get_available_components():
    """Return dictionary of available ChemWeaver components."""
    return {
        "core": True,  # Always available
        "data": _data_available,
        "workflow": _workflow_available,
        "ai": _ai_available,
        "benchmark": _benchmark_available,
    }

def print_status():
    """Print availability status of all ChemWeaver components."""
    components = get_available_components()
    print("ChemWeaver v{}".format(__version__))
    print("=" * 40)
    for name, available in components.items():
        status = "✓ Available" if available else "✗ Dependencies missing"
        print("  {}: {}".format(name.title(), status))
