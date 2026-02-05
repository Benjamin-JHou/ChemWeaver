"""
VSSS Minimal Package
====================

This package provides the core functionality for the VSSS minimal
virtual screening pipeline.

It includes:
- Compound and screening data structures
- AI surrogate model for score prediction
- Uncertainty quantification
- Workflow execution

"""

__version__ = "1.0.0"
__author__ = "VSSS Development Team"

from .core.pipeline import Compound, MinimalScreeningPipeline
from .core.inference import MinimalSurrogateModel, Prediction

__all__ = [
    "Compound",
    "MinimalScreeningPipeline",
    "MinimalSurrogateModel",
    "Prediction",
]
