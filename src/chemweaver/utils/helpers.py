"""
ChemWeaver Utilities
====================

Utility functions for the ChemWeaver minimal screening pipeline.

Author: ChemWeaver Development Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_reproducibility_hash(
    input_file: str,
    parameters: Dict[str, Any],
    timestamp: str
) -> str:
    """
    Compute reproducibility hash for a screening run.
    
    Args:
        input_file: Path to input file
        parameters: Screening parameters
        timestamp: Execution timestamp
        
    Returns:
        SHA-256 hash string
    """
    # Read input file
    input_content = Path(input_file).read_bytes()
    input_hash = hashlib.sha256(input_content).hexdigest()[:16]
    
    # Hash parameters
    param_str = json.dumps(parameters, sort_keys=True)
    param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    # Combine
    combined = f"{input_hash}:{param_hash}:{timestamp}"
    final_hash = hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    return final_hash


def validate_smiles(smiles: str) -> bool:
    """
    Basic SMILES validation.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if valid, False otherwise
    """
    if not smiles or len(smiles) < 2:
        return False
    
    # Check for balanced parentheses
    if smiles.count('(') != smiles.count(')'):
        return False
    
    if smiles.count('[') != smiles.count(']'):
        return False
    
    # Check for valid characters
    valid_chars = set('ABCDEFGHIKLMNOPRSTUVWXYZabcdefghiklmnopqrstuvwxyz0123456789=#()-[]/@.\\+%~')
    if not all(c in valid_chars for c in smiles):
        return False
    
    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)
