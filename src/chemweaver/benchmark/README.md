# VS-Bench: Virtual Screening Benchmark Ecosystem

**A Global, Open, Continuously Evolving Benchmark for AI-Assisted Virtual Screening**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/vs-bench/vs-bench)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-active-success.svg)](https://vs-bench.org)

---

## Mission

VS-Bench provides a comprehensive, FAIR-compliant benchmark ecosystem for evaluating AI-assisted virtual screening methods. Our goal is to:

- **Standardize evaluation** across the computational drug discovery community
- **Enable fair comparison** of virtual screening methods
- **Drive innovation** through comprehensive capability assessment
- **Ensure reproducibility** with rigorous protocols
- **Foster collaboration** through open community infrastructure

## Benchmark Suite Overview

### Six Capability Axes

VS-Bench evaluates methods across six critical dimensions:

| Axis | Description | Weight |
|------|-------------|--------|
| **Docking Surrogate** | Predict docking scores as surrogate for expensive calculations | 25% |
| **ADMET Prediction** | Predict absorption, distribution, metabolism, excretion, toxicity | 25% |
| **Multi-Target Activity** | Predict activity across multiple protein targets | 20% |
| **OOD Generalization** | Generalize to chemically novel scaffolds | 15% |
| **Cross-Protein Transfer** | Transfer across protein families | 15% |
| **Experimental Validation** | Validate on experimentally measured holdout set | Bonus |

### Sub-Benchmarks

The standard suite includes 6 sub-benchmarks:

1. **Docking Surrogate Main**: 150K training, 15K validation, 15K test (PDBbind-derived)
2. **Docking Surrogate OOD**: Scaffold-based OOD split for generalization testing
3. **ADMET Multi-Task**: 80K samples across 5 ADMET endpoints
4. **Multi-Target Activity**: 250K samples across 50 protein targets (ChEMBL)
5. **OOD Generalization**: 200K test set with novel scaffolds
6. **Cross-Protein Transfer**: Kinases, GPCRs, and proteases for family transfer
7. **Experimental Validation**: 5K hidden test set with experimental measurements

## Quick Start

### Installation

```bash
git clone https://github.com/vs-bench/vs-bench.git
cd vs-bench
pip install -e .
```

### Download Datasets

```python
from benchmark.dataset_engineering import DatasetBuilder

# Download pre-built datasets
builder = DatasetBuilder("vs-bench-standard", version="1.0.0")
dataset = builder.add_chembl_data(
    assay_id="all",
    min_confidence=8
).apply_scaffold_split(
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1
).set_license(
    license=LicenseType.CC_BY
).build()

# Export to Parquet
dataset.export_to_parquet("./data/")
```

### Evaluate Your Method

```python
from benchmark.evaluation_metrics import MetricsCalculator
import pandas as pd
import numpy as np

# Load your predictions and ground truth
predictions = pd.read_csv("your_predictions.csv")
ground_truth = pd.read_csv("ground_truth.csv")

# Merge and evaluate
merged = predictions.merge(ground_truth, on="compound_id")
y_true = merged["ground_truth"].values
y_pred = merged["predicted_value"].values

# Calculate metrics
reg_metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
rank_metrics = MetricsCalculator.calculate_ranking_metrics(y_true, y_pred)

print(f"Spearman: {reg_metrics.spearman:.3f}")
print(f"BEDROC: {rank_metrics.bedroc:.3f}")
print(f"EF@1%: {rank_metrics.enrichment_factor_1:.1f}")
```

### Submit to Leaderboard

```python
from benchmark.leaderboard_system import (
    SubmissionMetadata, SubmissionFiles, Leaderboard
)
from benchmark.scientific_design import create_standard_suite

# Create submission metadata
metadata = SubmissionMetadata(
    team_name="Your Team",
    method_name="Your Method",
    method_description="Graph neural network with attention",
    architecture="GNN",
    benchmark_suite="VS-Bench Standard"
)

# Define submission files
files = SubmissionFiles(
    predictions_csv="predictions.csv",
    metadata_json="metadata.json",
    environment_yaml="environment.yaml",
    run_script="run.sh"
)

# Submit to leaderboard
suite = create_standard_suite()
leaderboard = Leaderboard("VS-Bench Main", suite)
submission_id, result = leaderboard.submit(metadata, files)

print(f"Submission ID: {submission_id}")
print(f"Global Score: {result.global_score:.4f}")
print(f"Rank: {result.rank}")
```

## Dataset Schema

### Unified Benchmark Sample

```python
@dataclass
class BenchmarkSample:
    sample_id: UUID
    compound: MolecularIdentity      # SMILES, InChI, InChIKey
    protein: ProteinIdentity         # UniProt, sequence, family
    experimental_activity: float     # pIC50, pKi, etc.
    docking_score: float            # kcal/mol
    admet_properties: ADMETProperties
    assay_metadata: AssayMetadata
    split: str                      # train, val, test, hidden_test
    scaffold_id: str
```

### Data Card Example

Each dataset includes comprehensive metadata:

```yaml
Dataset: VS-Bench Standard v1.0.0
Sources:
  - ChEMBL 33
  - PDBbind 2020
  - BindingDB (updated 2023)

Statistics:
  Total samples: 665,000
  Unique compounds: 450,000
  Protein targets: 50
  Unique scaffolds: 180,000

Splits:
  Training: 80% (scaffold split)
  Validation: 10% (scaffold split)
  Test: 10% (scaffold split)
  Hidden Test: 5,000 (annual refresh)

Known Biases:
  - Enriched for drug-like chemical space
  - Underrepresentation of natural products
  - Activity cliff imbalance

License: CC-BY-4.0
DOI: 10.5281/zenodo.xxxxx
```

## Evaluation Metrics

### Regression Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Spearman ρ**: Rank correlation
- **Concordance Index**: Pairwise ranking accuracy
- **Top-K Recall**: Recall of top K compounds

### Ranking Metrics

- **AUC-ROC**: Area Under ROC Curve
- **BEDROC**: Boltzmann-Enhanced Discrimination of ROC
- **Enrichment Factor**: Fold enrichment at top X%
- **Precision@K**: Precision at rank K
- **Hit Rate**: Percentage of actives in top X%

### Calibration Metrics

- **ECE**: Expected Calibration Error
- **Brier Score**: Probabilistic calibration
- **Reliability Slope**: Calibration curve slope

### Robustness Metrics

- **OOD Performance Ratio**: OOD / In-distribution performance
- **Cross-Target Score**: Transfer learning capability
- **Uncertainty-Error Correlation**: Quality of uncertainty estimates

## Submission Format

### Required Files

```
submission/
├── predictions.csv       # Prediction file
├── metadata.json         # Method metadata
├── environment.yaml      # Conda environment
└── run.sh               # Reproduction script
```

### Predictions CSV

```csv
compound_id,predicted_value,uncertainty
CHEMBL12345,-8.23,0.45
CHEMBL67890,-7.12,0.38
...
```

### Metadata JSON

```json
{
  "team_name": "Your Team",
  "method_name": "GNN-Attn",
  "method_description": "Graph attention network for virtual screening",
  "architecture": "GNN",
  "training_time_hours": 48,
  "inference_time_per_compound_ms": 5.2,
  "hardware_used": "NVIDIA V100 32GB",
  "code_repository": "https://github.com/yourteam/gnn-attn",
  "docker_image": "yourteam/gnn-attn:latest",
  "benchmark_suite": "VS-Bench Standard",
  "benchmark_version": "1.0.0"
}
```

## Leaderboard

### Current Rankings

| Rank | Team | Method | Global Score | Docking | ADMET | Multi-Target | OOD | Robustness |
|------|------|--------|--------------|---------|-------|--------------|-----|------------|
| 1 | DeepChem | ChemBERTa | 0.823 | 0.84 | 0.81 | 0.79 | 0.76 | 0.78 |
| 2 | MIT | GraphMVP | 0.812 | 0.83 | 0.79 | 0.78 | 0.74 | 0.76 |
| 3 | Stanford | Mole-BERT | 0.805 | 0.82 | 0.78 | 0.77 | 0.73 | 0.75 |

*Leaderboard updated daily. See [full leaderboard](https://vs-bench.org/leaderboard)*

### Global Score Calculation

```python
Global Score = 
    0.25 × Docking Score +
    0.25 × ADMET Score +
    0.20 × Multi-Target Score +
    0.15 × OOD Generalization Score +
    0.15 × Robustness Score
```

## Community & Contributions

### Contribution Workflow

1. **Fork** the repository
2. **Add** your contribution to `submissions/<team-name>/`
3. **Submit** a Pull Request
4. **Automatic validation** runs on PR
5. **Community review** (7-day period)
6. **Merge** and leaderboard update

### Contribution Types

- **New Method**: Submit your model's predictions
- **New Task**: Propose new benchmark tasks
- **Bug Fixes**: Fix dataset or evaluation issues
- **Documentation**: Improve guides and examples

### Community Voting

Propose new benchmark tasks through GitHub Issues:
- Tasks require 10+ votes
- 70% approval threshold for acceptance
- Annual community review of benchmark suite

## Continuous Evolution

### Hidden Test Sets

- **5,000 compound hidden test set** for unbiased evaluation
- **Annual refresh** to prevent overfitting
- **Access logging** to detect data leakage

### Contamination Monitoring

- Statistical anomaly detection
- Compound overlap checking
- Perfect prediction flagging
- Automated investigation triggers

### Dataset Versioning

- Semantic versioning (Major.Minor.Patch)
- Version compatibility guarantees
- Complete change history
- Migration guides between versions

## Visualization & Analytics

### Available Dashboards

1. **Performance Distribution**: Violin plots across methods
2. **OOD Failure Analysis**: Per-category performance drops
3. **Calibration Plots**: Reliability diagrams with ECE
4. **Radar Charts**: Multi-dimensional method comparison
5. **Enrichment Curves**: ROC and enrichment visualization

### Generating Visualizations

```python
from benchmark.visualization import BenchmarkVisualizer

visualizer = BenchmarkVisualizer()

# Generate distribution plot
plot = visualizer.generate_performance_distribution(
    benchmark_scores={"docking": [0.7, 0.8, 0.75], "admet": [0.6, 0.7, 0.65]},
    metric="spearman"
)

# Export to Plotly
plotly_json = visualizer.to_plotly_json(plot)
```

## Reproducibility

### Environment Specification

All submissions require complete environment specification:

**Conda Environment**:
```yaml
name: vs-bench-env
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pytorch=2.0.1
  - rdkit=2023.09.1
  - pip
  - pip:
    - dgl==1.1.2
    - torch-geometric==2.3.1
```

**Dockerfile**:
```dockerfile
FROM rcvsi/ai-inference:1.0.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["./run.sh"]
```

### Random Seeds

All submissions must document random seeds:
```python
random_seeds = {
    "numpy": 42,
    "torch": 42,
    "python": 42,
    "dgl": 42
}
```

## Documentation

- **Scientific Design**: [benchmark/scientific_design.py](benchmark/scientific_design.py)
- **Dataset Engineering**: [benchmark/dataset_engineering.py](benchmark/dataset_engineering.py)
- **Evaluation Metrics**: [benchmark/evaluation_metrics.py](benchmark/evaluation_metrics.py)
- **API Reference**: https://vs-bench.readthedocs.io
- **Tutorials**: https://vs-bench.org/tutorials

## Citation

If you use VS-Bench in your research, please cite:

```bibtex
@article{vsbench2024,
  title={VS-Bench: A Comprehensive Benchmark for AI-Assisted Virtual Screening},
  author={[Authors]},
  journal={Nature Methods},
  year={2024},
  doi={10.xxxx/zenodo.vs-bench}
}
```

## License

VS-Bench is released under the MIT License. Datasets are released under CC-BY-4.0 unless otherwise specified.

## Contact

- **Website**: https://vs-bench.org
- **Issues**: https://github.com/vs-bench/vs-bench/issues
- **Discussions**: https://github.com/vs-bench/vs-bench/discussions
- **Email**: contact@vs-bench.org

## Acknowledgments

- ChEMBL team for bioactivity data
- RDKit community for cheminformatics tools
- Open source ML community
- Contributing research groups worldwide

---

**Made with ❤️ by the VS-Bench Community**

Part of the Virtual Screening Standard Schema (VSSS) ecosystem.
