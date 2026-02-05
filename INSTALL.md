# Installation

## Option 1: Clone and install locally
```bash
git clone https://github.com/Benjamin-JHou/ChemWeaver.git
cd ChemWeaver
pip install -e .
```

## Option 2: Install from PyPI (when published)
```bash
pip install chemweaver
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