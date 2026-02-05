# ChemWeaver Installation

## 🚀 Quick Start (Recommended)

### One-Click Deployment
```bash
git clone https://github.com/Benjamin-JHou/ChemWeaver.git
cd ChemWeaver
./deploy_chemweaver.sh
```

### Manual Installation
```bash
git clone https://github.com/Benjamin-JHou/ChemWeaver.git
cd ChemWeaver
pip install -e .
```

## 📦 Full Installation (All Components)

For complete functionality with AI models and workflow engines:
```bash
git clone https://github.com/Benjamin-JHou/ChemWeaver.git
cd ChemWeaver
./install_dependencies.sh
pip install -e .
```

## 📋 Component Availability

ChemWeaver uses smart dependency loading:

- ✅ **Core**: Always available (basic screening + AI)
- ✅ **Data**: Standardized data formats & storage  
- ✅ **Benchmark**: Evaluation framework
- ⚡ **Workflow**: Compute-adaptive engines (requires h5py, pyarrow)
- 🤖 **AI**: Full multi-modal models (requires torch, transformers)

Check your current status:
```bash
python3 -c "import chemweaver; chemweaver.print_status()"
```

# Quick Start

## Run with example data
```bash
chemweaver --example
```

## Run with your own data
```bash
chemweaver -i your_compounds.smi -o results.json
```

## Using as Python library
```python
from chemweaver import Compound, MinimalScreeningPipeline
from chemweaver.core.inference import MinimalSurrogateModel

# Create compound from SMILES
compound = Compound.from_smiles("CCO", "compound_1")

# Initialize pipeline
pipeline = MinimalScreeningPipeline()

# Run screening
results = pipeline.run_screening([compound])
```

# Requirements
- Python 3.8+
- Dependencies listed in requirements.txt

# Docker Usage
```bash
docker build -t chemweaver .
docker run -v $(pwd)/data:/app/data chemweaver --example
```