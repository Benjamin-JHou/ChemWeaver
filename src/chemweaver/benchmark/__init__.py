"""
ChemWeaver Benchmark Module

This module provides comprehensive benchmarking capabilities for ChemWeaver,
including dataset engineering, evaluation metrics, and visualization tools.

Author: ChemWeaver Development Team
Version: 1.0.0
"""

# Import benchmark modules dynamically to handle missing dependencies
try:
    from .dataset_engineering import *
    _DATASET_AVAILABLE = True
except ImportError:
    _DATASET_AVAILABLE = False

try:
    from .evaluation_metrics import *
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

try:
    from .visualization import *
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

try:
    from .leaderboard_system import *
    _LEADERBOARD_AVAILABLE = True
except ImportError:
    _LEADERBOARD_AVAILABLE = False

__all__ = [
    # Dataset engineering
    "DatasetEngineer",
    "BenchmarkDataset",
    
    # Evaluation metrics
    "EnrichmentMetrics",
    "CalibrationMetrics",
    
    # Visualization
    "BenchmarkVisualizer",
    
    # Leaderboard
    "LeaderboardSystem",
]

# Availability flags
__version__ = "1.0.0"
_dataset_available = _DATASET_AVAILABLE
_metrics_available = _METRICS_AVAILABLE
_viz_available = _VIZ_AVAILABLE
_leaderboard_available = _LEADERBOARD_AVAILABLE