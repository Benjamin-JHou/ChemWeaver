"""
ChemWeaver AI Module

This module provides the AI surrogate models with uncertainty quantification
for ChemWeaver, including multi-modal neural networks and decision systems.

Author: ChemWeaver Development Team
Version: 1.0.0
"""

from .models.surrogate import *
from .uncertainty.quantification import *
from .inference.decision_layer import *

__all__ = [
    # AI models
    "ChemWeaverSurrogate", 
    "ModelConfig",
    
    # Uncertainty quantification
    "UncertaintyQuantifier",
    "CalibrationMetrics",
    
    # Decision layer
    "DecisionLayer",
    "HitSelector",
    
    # Architecture components
    "MultiModalNetwork",
    "PhysicsRegularizedLoss",
]