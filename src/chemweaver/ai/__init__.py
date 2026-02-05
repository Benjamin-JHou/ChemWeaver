"""
ChemWeaver AI Module

This module provides the AI surrogate models with uncertainty quantification
for ChemWeaver, including multi-modal neural networks and decision systems.

Author: ChemWeaver Development Team
Version: 1.0.0
"""

# Smart dependency loading
from ..utils.dependencies import deps

# Check AI dependencies
ai_deps = deps.get_ai_status()
has_full_ai = all(ai_deps.values())

if has_full_ai:
    # Full AI functionality available
    try:
        from .models.surrogate import *
        from .uncertainty.quantification import *
        from .inference.decision_layer import *
        
        _FULL_AI_AVAILABLE = True
        print("✅ ChemWeaver-AI: Full functionality available")
        
    except Exception as e:
        print(f"⚠️ ChemWeaver-AI: Import error - {e}")
        _FULL_AI_AVAILABLE = False
else:
    # Missing dependencies - provide fallback
    print("⚠️ ChemWeaver-AI: Limited functionality due to missing dependencies")
    missing = [k for k, v in ai_deps.items() if not v]
    print(f"   Missing: {', '.join(missing)}")
    print("   Install with: pip install torch torch-geometric transformers")
    _FULL_AI_AVAILABLE = False

# Always provide minimal AI interface
from ..core.inference import MinimalSurrogateModel, Prediction

__all__ = [
    # Always available
    "MinimalSurrogateModel",
    "Prediction",
]

# Full AI exports (when available)
if _FULL_AI_AVAILABLE:
    __all__.extend([
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
    ])

# Status flag
_full_ai_available = _FULL_AI_AVAILABLE