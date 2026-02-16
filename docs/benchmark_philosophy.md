# ChemWeaver Benchmark Philosophy

## Scientific Rigor in Method Evaluation

### The Benchmarking Crisis

Virtual screening methods are often evaluated using inconsistent or biased benchmarks, leading to inflated performance claims and difficulty in comparing methods. ChemWeaver addresses this through:

1. **Standardized Evaluation Protocols**
2. **Fair Comparison Frameworks**
3. **Transparent Reporting Standards**
4. **Community-Driven Benchmarks**

---

## Core Principles

### 1. Reproducibility First

**Principle**: Every benchmark result must be fully reproducible

**Implementation**:
```python
# Complete benchmark specification
benchmark_config = {
    "dataset": "DUD-E",
    "version": "2016",
    "split": "scaffold",
    "metrics": ["AUC-ROC", "AUC-PR", "EF@1%"],
    "random_seed": 42,
    "software_versions": {
        "chemweaver": "1.0.0",
        "rdkit": "2023.09.1"
    }
}

# Save with results
results["benchmark_config"] = benchmark_config
results["reproducibility_hash"] = compute_hash(benchmark_config)
```

### 2. Fair Data Splits

**Principle**: Prevent data leakage through proper splitting strategies

**Supported Splits**:

| Split Type | Method | Use Case |
|------------|--------|----------|
| **Scaffold** | Bemis-Murcko scaffold | Generalization assessment |
| **Temporal** | Publication date | Real-world deployment |
| **Protein Family** | Target class | Transfer learning |
| **Chemical OOD** | Cluster-based | Novelty detection |

**Example**:
```python
# Scaffold split - most rigorous for drug discovery
from chemweaver.benchmark import scaffold_split

train, val, test = scaffold_split(
    compounds,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15
)

# Ensures no scaffold overlap between splits
assert no_scaffold_overlap(train, test)
```

### 3. Comprehensive Metrics

**Principle**: Evaluate multiple aspects of model performance

**Metric Categories**:

**A. Ranking Metrics**
- AUC-ROC: Overall discrimination
- AUC-PR: Performance on imbalanced data
- BEDROC (α=20): Early enrichment emphasis
- Enrichment Factor (EF@1%, 5%, 10%)

**B. Calibration Metrics**
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams

**C. Uncertainty Metrics**
- Uncertainty-error correlation
- Coverage probability
- Sharpness

**D. Robustness Metrics**
- OOD detection AUC
- Performance under perturbation
- Cross-target transfer

### 4. Appropriate Baselines

**Principle**: Compare against relevant, strong baselines

**Required Baselines**:

| Method Type | Baselines |
|-------------|-----------|
| **Docking Surrogate** | AutoDock Vina, Glide SP, RF-Score, NNScore |
| **ML Models** | Random Forest, XGBoost, Feed-forward NN |
| **GNN Models** | GCN, GAT, D-MPNN |
| **Uncertainty** | MC Dropout, Deep Ensembles, Evidential |

### 5. Statistical Significance

**Principle**: Report confidence intervals and significance tests

**Implementation**:
```python
from scipy import stats

# Compute confidence intervals
auc_scores = [run_experiment(seed) for seed in range(10)]
ci_low, ci_high = np.percentile(auc_scores, [2.5, 97.5])

# Statistical significance test
stat, p_value = stats.wilcoxon(
    our_method_scores,
    baseline_scores
)
significant = p_value < 0.05
```

---

## Benchmark Datasets

### Primary Benchmarks

**1. DUD-E (Database of Useful Decoys - Enhanced)**
- 102 targets
- 22,886 active compounds
- 1,285,357 decoys
- Standard for virtual screening evaluation

**Usage**:
```python
from chemweaver.benchmark import load_dude_benchmark

benchmark = load_dude_benchmark(
    target="aa2ar",
    split="scaffold"
)
```

**2. MUV (Maximum Unbiased Validation)**
- 17 challenging targets
- Designed to avoid artificial enrichment
- More difficult than DUD-E

**3. ChEMBL Bioactivity**
- 280,000+ bioactivity measurements
- Multiple target classes
- Real-world experimental data

**4. PDBbind**
- 19,000 protein-ligand complexes
- Binding affinity data
- Structural information

### Benchmark Suite: ChemWeaver-Bench

**Design Goals**:
1. **Comprehensive**: Cover multiple aspects of VS
2. **Fair**: No data leakage, proper splits
3. **Challenging**: Realistic difficulty
4. **Evolving**: Regular updates to prevent overfitting

**Tasks**:

| Task | Dataset | Metric | Weight |
|------|---------|--------|--------|
| Docking Prediction | PDBbind | Spearman ρ | 25% |
| Activity Classification | ChEMBL | AUC-ROC | 20% |
| ADMET Prediction | ADMET Lab | RMSE | 25% |
| OOD Generalization | Scaffold split | OOD/ID ratio | 15% |
| Cross-Target Transfer | Kinase→GPCR | Transfer score | 15% |

---

## Evaluation Protocols

### Protocol 1: Retrospective Benchmarking

**Purpose**: Standard method comparison on historical data

**Steps**:
1. Load benchmark dataset with specified split
2. Train model on training set
3. Validate on validation set (hyperparameter tuning)
4. Test on held-out test set (single evaluation)
5. Report metrics with confidence intervals

**Requirements**:
- No test set access during development
- Single test evaluation per method version
- Report all metrics, not just best ones

### Protocol 2: Prospective Validation

**Purpose**: Real-world validation with experimental follow-up

**Steps**:
1. Pre-register study protocol
2. Screen library with locked parameters
3. Select hits using defined criteria
4. Experimental validation
5. Report hit rates and validation success

**Standards**:
- Pre-registration mandatory
- No post-hoc tuning
- Full protocol disclosure
- Cost transparency

### Protocol 3: Ablation Studies

**Purpose**: Understand component contributions

**Design**:
```python
ablation_configs = [
    {"name": "Full Model", "components": ["GNN", "Transformer", "UQ"]},
    {"name": "No GNN", "components": ["Transformer", "UQ"]},
    {"name": "No Transformer", "components": ["GNN", "UQ"]},
    {"name": "No UQ", "components": ["GNN", "Transformer"]},
]

for config in ablation_configs:
    model = build_model(config["components"])
    results = evaluate(model, benchmark)
    report(results)
```

---

## Reporting Standards

### Required Information

Every benchmark report must include:

**1. Dataset Information**
```yaml
Dataset: DUD-E
Version: 2016
Targets: 102
Actives: 22,886
Decoys: 1,285,357
Split: Scaffold-based
Train/Val/Test: 70/15/15%
```

**2. Model Configuration**
```yaml
Model: ChemWeaver-AI-Surrogate
Architecture: D-MPNN + Transformer
Parameters: 5.2M
Training Data: PDBbind 2020
Random Seed: 42
```

**3. Evaluation Metrics**
```yaml
Primary: AUC-ROC
Secondary: [AUC-PR, EF@1%, BEDROC]
Calibration: ECE, Brier Score
Uncertainty: Correlation, Coverage
```

**4. Computational Resources**
```yaml
CPU: Intel Xeon E5-2680 v4
GPU: NVIDIA V100
Memory: 128GB
Training Time: 24 hours
Inference Time: 0.05s/compound
```

### Results Table Template

| Target | AUC-ROC | AUC-PR | EF@1% | ECE |
|--------|---------|--------|-------|-----|
| ABL1 | 0.89±0.02 | 0.45±0.03 | 18.2±1.5 | 0.042±0.005 |
| AA2AR | 0.87±0.03 | 0.42±0.04 | 17.8±1.8 | 0.038±0.006 |
| **Mean** | **0.88±0.02** | **0.43±0.03** | **18.0±1.6** | **0.040±0.005** |

### Statistical Reporting

**Confidence Intervals**:
- Report 95% confidence intervals for all metrics
- Computed via bootstrap (1000 iterations) or cross-validation

**Significance Testing**:
- Use Wilcoxon signed-rank test for paired comparisons
- Report p-values
- Set significance threshold at p < 0.05

**Effect Sizes**:
- Report Cohen's d for practical significance
- d < 0.2: negligible
- 0.2 ≤ d < 0.5: small
- 0.5 ≤ d < 0.8: medium
- d ≥ 0.8: large

---

## Community Benchmarks

### ChemWeaver Leaderboard

**Purpose**: Fair, ongoing method comparison

**Features**:
- Automated submission validation
- Standardized evaluation
- Live leaderboard
- Version tracking

**Submission Process**:
1. Fork repository
2. Add method to `submissions/`
3. Submit pull request
4. Automated evaluation
5. Results merged to leaderboard

**Leaderboard Schema**:
```json
{
  "rank": 1,
  "method": "ChemWeaver-Ensemble",
  "authors": "Hou et al.",
  "institution": "Independent Submission",
  "metrics": {
    "docking_spearman": 0.89,
    "activity_auc": 0.87,
    "admet_rmse": 0.82,
    "ood_ratio": 0.85
  },
  "global_score": 0.89,
  "submission_date": "2024-02-01",
  "code_available": true,
  "preprint": "pending"
}
```

### Continuous Benchmarking

**Challenge**: Prevent overfitting to static benchmarks

**Solution**: Rotating test sets

**Implementation**:
- Hidden test set refreshed annually
- Previous test set becomes public
- Access logged to detect contamination

**Timeline**:
```
2024-Q1: Release ChemWeaver-Bench v1.0
2025-Q1: Release v1.1 with refreshed hidden test
2026-Q1: Release v1.2 with new targets
```

---

## Best Practices

### For Method Developers

1. **Use Proper Splits**
   - Never random split for molecular data
   - Always use scaffold or temporal splits
   - Verify no data leakage

2. **Report All Results**
   - Don't cherry-pick best targets
   - Report failures as well as successes
   - Include negative results

3. **Provide Code**
   - Open source implementation
   - Docker container for reproduction
   - Clear documentation

4. **Validate on Multiple Benchmarks**
   - Don't optimize for single benchmark
   - Test on diverse target classes
   - Include OOD evaluation

### For Benchmark Users

1. **Understand Limitations**
   - Benchmarks are approximations
   - Real-world performance may vary
   - Consider multiple benchmarks

2. **Check for Data Leakage**
   - Verify split methodology
   - Check for duplicate compounds
   - Confirm temporal splits where appropriate

3. **Consider Computational Cost**
   - Inference time matters
   - Training resources should be reported
   - Scalability is important

---

## Limitations and Future Work

### Current Limitations

1. **Benchmark Bias**
   - Historical benchmarks may not represent current chemical space
   - Target selection may favor certain methods
   - Decoy generation has known issues

2. **Metric Limitations**
   - AUC-ROC can be misleading for imbalanced data
   - Enrichment factors depend on library composition
   - Calibration metrics don't capture all uncertainty aspects

3. **Real-World Gap**
   - Benchmarks don't capture all real-world challenges
   - Assay variability not modeled
   - Chemical synthesis feasibility ignored

### Future Directions

1. **Prospective Benchmarks**
   - Real experimental validation
   - Cost-aware evaluation
   - Synthetic accessibility scoring

2. **Dynamic Benchmarks**
   - Continuously updated test sets
   - Active learning integration
   - Community-contributed targets

3. **Multi-Objective Benchmarks**
   - Activity + selectivity
   - Activity + ADMET
   - Multi-target optimization

---

## References

### Benchmark Datasets

1. **DUD-E**
   - Mysinger et al., J. Chem. Inf. Model. 2012
   - http://dude.docking.org

2. **MUV**
   - Rohrer & Baumann, J. Chem. Inf. Model. 2009
   - Designed to avoid artificial enrichment

3. **ChEMBL**
   - Mendez et al., Nucleic Acids Res. 2019
   - https://www.ebi.ac.uk/chembl

4. **PDBbind**
   - Wang et al., J. Med. Chem. 2004
   - http://www.pdbbind.org.cn

### Methodology Papers

1. **Scaffold Splitting**
   - Bemis & Murcko, J. Med. Chem. 1996
   - Wu et al., J. Chem. Inf. Model. 2018

2. **Evaluation Metrics**
   - Truchon & Bayly, J. Chem. Inf. Model. 2007 (BEDROC)
   - Nicholls, J. Comput. Aided Mol. Des. 2014

3. **Uncertainty Quantification**
   - Gal & Ghahramani, ICML 2016 (MC Dropout)
   - Lakshminarayanan et al., NIPS 2017 (Deep Ensembles)

---

## Contact

For benchmark questions or contributions:
- GitHub Issues: [ChemWeaver Issues](https://github.com/Benjamin-JHou/ChemWeaver/issues)
- Repository: https://github.com/Benjamin-JHou/ChemWeaver

---

**Version**: 1.0.0  
**Last Updated**: 2026-02-05  
**Maintainer**: ChemWeaver Development Team
