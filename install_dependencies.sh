#!/bin/bash

# ChemWeaver Complete Dependencies Installer
# =========================================

set -e

echo "🔬 ChemWeaver Dependencies Installer"
echo "=================================="
echo ""

# Check available components
echo "📊 Checking current ChemWeaver status..."
python3 -c "
import sys
sys.path.insert(0, 'src')
import chemweaver
chemweaver.print_status()
"
echo ""

# Function to install with fallback
install_with_fallback() {
    local package=$1
    local fallback=$2
    
    echo "📦 Installing $package..."
    if pip install "$package"; then
        echo "✅ $package installed successfully"
    else
        echo "⚠️ Failed to install $package, trying fallback: $fallback"
        pip install "$fallback" || echo "❌ Failed to install both versions"
    fi
}

# Install core dependencies (always needed)
echo "🔧 Installing core dependencies..."
pip install --upgrade pip
install_with_fallback "torch>=1.12.0" "torch"
install_with_fallback "torch-geometric>=2.0.0" "torch-geometric"
install_with_fallback "transformers>=4.15.0" "transformers"
install_with_fallback "h5py>=3.6.0" "h5py"
install_with_fallback "pyarrow>=6.0.0" "pyarrow"
install_with_fallback "jsonschema>=4.0.0" "jsonschema"

# Optional but recommended dependencies
echo ""
echo "🎯 Installing optional but recommended dependencies..."
install_with_fallback "nextflow>=22.04.0" "nextflow"
install_with_fallback "plotly>=5.0.0" "plotly"
install_with_fallback "jupyter>=1.0.0" "jupyter"

# GPU support (optional)
echo ""
echo "🚀 Checking GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected - installing CUDA support"
    install_with_fallback "torch-scatter" "torch-scatter"
    install_with_fallback "torch-sparse" "torch-sparse"
    install_with_fallback "torch-cluster" "torch-cluster"
else
    echo "ℹ️ No NVIDIA GPU detected - using CPU only (still fully functional)"
fi

echo ""
echo "🧪 Testing final installation..."
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    import chemweaver
    chemweaver.print_status()
    print('✅ ChemWeaver installation complete!')
except Exception as e:
    print(f'❌ Installation error: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Installation complete! ChemWeaver is ready for use."
echo ""
echo "📚 Quick start:"
echo "   chemweaver --example"
echo "   chemweaver -i your_compounds.smi -o results.json"
echo ""