#!/bin/bash

# ChemWeaver One-Click Deployment Script
# =====================================

set -e

echo "🧪 ChemWeaver One-Click Deployment"
echo "================================="
echo "Installing comprehensive virtual screening infrastructure..."
echo ""

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
echo "📋 Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]] 2>/dev/null; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "chemweaver_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv chemweaver_env
fi

echo "🔄 Activating virtual environment..."
source chemweaver_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install ChemWeaver in development mode
echo "🧬 Installing ChemWeaver..."
pip install -e .

echo ""
echo "🔍 Testing installation..."
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    import chemweaver
    chemweaver.print_status()
    print('✅ ChemWeaver installation successful!')
except Exception as e:
    print(f'❌ Installation error: {e}')
    sys.exit(1)
"

echo ""
echo "🚀 Quick Start Guide:"
echo "===================="
echo "1. Run with example data:"
echo "   chemweaver --example"
echo ""
echo "2. Run with your own data:"
echo "   chemweaver -i your_compounds.smi -o results.json"
echo ""
echo "3. Use as Python library:"
echo "   python3 -c \"from chemweaver import Compound, MinimalScreeningPipeline\""
echo ""
echo "4. Check available components:"
echo "   python3 -c \"import chemweaver; chemweaver.print_status()\""
echo ""

echo "✅ ChemWeaver deployment complete!"
echo "📚 Documentation: README.md, INSTALL.md"
echo "🐛 Issues: https://github.com/Benjamin-JHou/ChemWeaver/issues"
echo ""