"""
ChemWeaver Dependency Management

This module provides smart dependency loading for optional components,
allowing graceful degradation when heavy dependencies are unavailable.
"""

import sys
from typing import Optional, Dict, Any

class DependencyManager:
    """Manages optional dependencies for ChemWeaver components."""
    
    def __init__(self):
        self._dependency_cache: Dict[str, bool] = {}
    
    def check_torch(self) -> bool:
        """Check if PyTorch is available."""
        if 'torch' not in self._dependency_cache:
            try:
                import torch
                self._dependency_cache['torch'] = True
                print("✅ PyTorch available")
            except ImportError:
                self._dependency_cache['torch'] = False
                print("⚠️ PyTorch not available - Install with: pip install torch")
        return self._dependency_cache['torch']
    
    def check_torch_geometric(self) -> bool:
        """Check if PyTorch Geometric is available."""
        if 'torch_geometric' not in self._dependency_cache:
            try:
                import torch_geometric
                self._dependency_cache['torch_geometric'] = True
                print("✅ PyTorch Geometric available")
            except ImportError:
                self._dependency_cache['torch_geometric'] = False
                print("⚠️ PyTorch Geometric not available - Install with: pip install torch-geometric")
        return self._dependency_cache['torch_geometric']
    
    def check_transformers(self) -> bool:
        """Check if Transformers library is available."""
        if 'transformers' not in self._dependency_cache:
            try:
                import transformers
                self._dependency_cache['transformers'] = True
                print("✅ Transformers available")
            except ImportError:
                self._dependency_cache['transformers'] = False
                print("⚠️ Transformers not available - Install with: pip install transformers")
        return self._dependency_cache['transformers']
    
    def check_h5py(self) -> bool:
        """Check if h5py is available."""
        if 'h5py' not in self._dependency_cache:
            try:
                import h5py
                self._dependency_cache['h5py'] = True
                print("✅ h5py available")
            except ImportError:
                self._dependency_cache['h5py'] = False
                print("⚠️ h5py not available - Install with: pip install h5py")
        return self._dependency_cache['h5py']
    
    def check_pyarrow(self) -> bool:
        """Check if PyArrow is available."""
        if 'pyarrow' not in self._dependency_cache:
            try:
                import pyarrow
                self._dependency_cache['pyarrow'] = True
                print("✅ PyArrow available")
            except ImportError:
                self._dependency_cache['pyarrow'] = False
                print("⚠️ PyArrow not available - Install with: pip install pyarrow")
        return self._dependency_cache['pyarrow']
    
    def get_ai_status(self) -> Dict[str, bool]:
        """Get status of all AI-related dependencies."""
        return {
            'torch': self.check_torch(),
            'torch_geometric': self.check_torch_geometric(),
            'transformers': self.check_transformers(),
        }
    
    def get_data_status(self) -> Dict[str, bool]:
        """Get status of all data-related dependencies."""
        return {
            'h5py': self.check_h5py(),
            'pyarrow': self.check_pyarrow(),
        }

# Global dependency manager
deps = DependencyManager()