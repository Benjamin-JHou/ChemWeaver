# ChemWeaver Architecture

## System Design Philosophy

ChemWeaver implements a **layered, modular architecture** designed for:
- **Reproducibility**: Every component is deterministic and verifiable
- **Extensibility**: Easy to add new methods and integrate with existing tools
- **Scalability**: From laptop to HPC cluster
- **Transparency**: All operations are logged and auditable

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ChemWeaver Platform                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Interface Layer                                    │
│  ├─ Command Line Interface (CLI)                            │
│  ├─ Python API                                              │
│  └─ Container Interface (Docker/Singularity)                │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Workflow Orchestration                            │
│  ├─ Pipeline Controller                                     │
│  ├─ Stage Manager                                           │
│  ├─ Resource Monitor                                        │
│  └─ Execution Logger                                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Core Processing Engines                           │
│  ├─ Standardization Engine                                  │
│  ├─ Property Filter Engine                                  │
│  ├─ AI Surrogate Engine                                     │
│  │   ├─ Feature Extraction                                  │
│  │   ├─ Model Inference                                     │
│  │   └─ Uncertainty Quantification                          │
│  └─ Decision Engine                                         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Data & Infrastructure                             │
│  ├─ Data Standard (ChemWeaver Standard)                     │
│  ├─ Storage Layer (Parquet + HDF5)                          │
│  ├─ Reproducibility Layer                                   │
│  └─ Provenance Tracking                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Data & Infrastructure

### 1.1 ChemWeaver Standard (Data Model)

**Core Entities**:

```python
@dataclass
class Compound:
    """Chemical compound representation"""
    compound_id: str          # Unique identifier
    smiles: str              # Canonical SMILES
    inchikey: str            # InChIKey for integrity
    molecular_weight: float
    logp: float
    hbd: int                # Hydrogen bond donors
    hba: int                # Hydrogen bond acceptors
    
@dataclass
class ScreeningResult:
    """Result of screening a compound"""
    compound_id: str
    predicted_score: float       # Binding affinity
    uncertainty: float           # Prediction uncertainty
    confidence: float            # Calibrated confidence
    passed_filter: bool          # Selection status
    selection_reason: str        # Rationale
```

**Design Principles**:
1. **Immutability**: Core entities are immutable once created
2. **Hashability**: All entities can be hashed for integrity checks
3. **Serialization**: Native JSON support for persistence
4. **Validation**: Built-in validation for all fields

### 1.2 Storage Layer

**Hybrid Storage Architecture**:

```
data/
├── metadata/                    # Apache Parquet
│   ├── compounds.parquet       # Tabular metadata
│   └── results.parquet         # Screening results
├── structures/                  # HDF5
│   └── conformers.h5          # 3D coordinates
├── indexes/                     # Search indexes
│   └── fingerprints.index     # Similarity search
└── provenance/                  # Lineage
    └── execution_graph.json   # W3C PROV-O
```

**Performance**:
- Query: 1M+ rows/sec (Parquet)
- I/O: 100K+ atoms/sec (HDF5)
- Scale: 10^9+ compounds supported

### 1.3 Reproducibility Layer

**Components**:

```python
class ReproducibilityManager:
    """Manages reproducibility across all operations"""
    
    def __init__(self):
        self.seed = 42
        self.container_digest = None
        self.dependency_versions = {}
    
    def set_seed(self, seed: int):
        """Set random seed for all operations"""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)  # If using PyTorch
    
    def record_environment(self):
        """Record complete environment state"""
        return {
            "python": sys.version,
            "packages": get_package_versions(),
            "container": get_container_digest(),
            "timestamp": datetime.utcnow().isoformat()
        }
```

### 1.4 Provenance Tracking

**W3C PROV-O Implementation**:

```python
class ProvenanceTracker:
    """Tracks data lineage using W3C PROV-O standard"""
    
    def log_operation(
        self,
        activity: str,
        inputs: List[str],
        outputs: List[str],
        parameters: Dict
    ):
        """Log an operation with complete provenance"""
        record = {
            "activity": activity,
            "used": inputs,           # Input entities
            "generated": outputs,     # Output entities
            "started_at": timestamp,
            "ended_at": timestamp,
            "parameters": parameters
        }
        self.provenance_graph.add(record)
```

---

## Layer 2: Core Processing Engines

### 2.1 Standardization Engine

**Purpose**: Ensure consistent molecular representation

**Pipeline**:
```
Raw SMILES
    │
    ▼
┌─────────────────┐
│  Parse SMILES   │  RDKit validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Strip Salts     │  Remove counterions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Neutralize      │  pH-dependent protonation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Canonicalize    │  Tautomer enumeration
└────────┬────────┘
         │
         ▼
Standardized Compound
```

**Implementation**:
```python
class StandardizationEngine:
    """Standardizes molecular representations"""
    
    def standardize(self, compound: Compound) -> Compound:
        # Step 1: Parse and validate
        mol = self.parse_smiles(compound.smiles)
        
        # Step 2: Remove salts
        mol = self.remove_salts(mol)
        
        # Step 3: Neutralize charges
        mol = self.neutralize(mol, ph=7.4)
        
        # Step 4: Canonicalize tautomers
        mol = self.canonicalize_tautomers(mol)
        
        # Step 5: Generate canonical SMILES
        canonical_smiles = self.to_smiles(mol)
        
        return Compound.from_smiles(canonical_smiles)
```

### 2.2 Property Filter Engine

**Purpose**: Apply drug-likeness filters

**Filters**:

| Filter | Criteria | Pass Rate |
|--------|----------|-----------|
| **Lipinski** | MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10 | ~85% |
| **Veber** | RotB ≤ 10, TPSA ≤ 140 | ~90% |
| **Synthetic** | SA score < 6.0 | ~70% |
| **PAINS** | No flagged substructures | ~95% |

**Implementation**:
```python
class PropertyFilterEngine:
    """Filters compounds by drug-like properties"""
    
    def __init__(self, config: FilterConfig):
        self.filters = [
            LipinskiFilter(),
            VeberFilter(),
            SyntheticAccessibilityFilter(),
            PAINSFilter() if config.filter_pains else None
        ]
    
    def filter(self, compounds: List[Compound]) -> List[Compound]:
        passed = compounds
        for filter in self.filters:
            if filter:
                passed = [c for c in passed if filter.check(c)]
        return passed
```

### 2.3 AI Surrogate Engine

**Purpose**: Predict binding affinity without docking

**Architecture**:

```
Input: Compound SMILES
    │
    ├──┬──┬──┐
    │  │  │  │
    ▼  ▼  ▼  ▼
┌────┐┌────┐┌────┐┌────┐
│GNN ││Tran││Desc││Pocket│
│Enc ││Enc ││Enc ││Enc   │
└─┬──┘└─┬──┘└─┬──┘└──┬───┘
  │     │     │      │
  └─────┴──┬──┴──────┘
           │
           ▼
    ┌──────────────┐
    │ Fusion Layer │  Cross-modal attention
    │  (512-dim)   │
    └──────┬───────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│ Docking │  │ Activity│
│  Score  │  │   Prob  │
└─────────┘  └─────────┘
     │           │
     └─────┬─────┘
           │
           ▼
    ┌──────────────┐
    │  Uncertainty │
    │   Head(s)    │
    └──────────────┘
```

**Components**:

1. **GNN Encoder** (Molecular Graph)
   - Input: Atom features (74-dim), Bond features (10-dim)
   - Architecture: D-MPNN (4 layers, 256-dim)
   - Output: Graph embedding

2. **Transformer Encoder** (SMILES)
   - Input: SMILES tokens
   - Architecture: 6 layers, 8 heads, 256-dim
   - Output: Sequence embedding

3. **Descriptor Encoder** (Physicochemical)
   - Input: Morgan fingerprint (2048-bit)
   - Architecture: MLP (2048 → 512 → 256)
   - Output: Descriptor embedding

4. **Fusion Layer**
   - Cross-modal attention
   - Joint representation: 512-dim

5. **Task Heads**
   - Docking score: Linear(512 → 1)
   - Activity probability: Linear(512 → 1)

6. **Uncertainty Quantification**
   - Deep Ensemble: 5 models
   - MC Dropout: 20 samples
   - Evidential: Learned variance

**Implementation**:
```python
class AISurrogateEngine:
    """AI surrogate model for binding affinity prediction"""
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.feature_extractor = FeatureExtractor()
    
    def predict(
        self,
        compound: Compound,
        return_uncertainty: bool = True
    ) -> Prediction:
        # Extract features
        features = self.feature_extractor.extract(compound)
        
        # Predict score
        score = self.model.predict_score(features)
        
        if return_uncertainty:
            # Quantify uncertainty
            uncertainty = self.model.estimate_uncertainty(features)
            confidence = 1.0 - uncertainty
            
            return Prediction(
                compound_id=compound.compound_id,
                predicted_score=score,
                uncertainty=uncertainty,
                confidence=confidence,
                method="ensemble"
            )
        
        return Prediction(
            compound_id=compound.compound_id,
            predicted_score=score,
            uncertainty=0.0,
            confidence=1.0,
            method="point"
        )
```

### 2.4 Decision Engine

**Purpose**: Make risk-aware hit selection decisions

**Logic**:
```python
class DecisionEngine:
    """Makes uncertainty-aware selection decisions"""
    
    def evaluate(
        self,
        prediction: Prediction,
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.5
    ) -> Decision:
        
        # Check confidence
        if prediction.confidence < confidence_threshold:
            return Decision(
                status="FAIL",
                reason=f"Low confidence: {prediction.confidence:.2f}",
                action="Filter out or collect more data"
            )
        
        # Check uncertainty
        if prediction.uncertainty > uncertainty_threshold:
            return Decision(
                status="REVIEW",
                reason=f"High uncertainty: {prediction.uncertainty:.2f}",
                action="Manual review recommended"
            )
        
        # Pass all checks
        return Decision(
            status="PASS",
            reason="High confidence, low uncertainty",
            action="Select for experimental validation"
        )
```

---

## Layer 3: Workflow Orchestration

### 3.1 Pipeline Controller

**Purpose**: Coordinate multi-stage execution

```python
class PipelineController:
    """Orchestrates the complete screening workflow"""
    
    def __init__(self, config: PipelineConfig):
        self.stages = [
            StandardizationStage(),
            PropertyFilterStage(),
            AIScoringStage(),
            HitSelectionStage()
        ]
        self.logger = ExecutionLogger()
    
    def execute(
        self,
        compounds: List[Compound]
    ) -> List[ScreeningResult]:
        
        data = compounds
        
        for stage in self.stages:
            # Log stage start
            self.logger.log_stage_start(stage.name)
            
            # Execute stage
            data = stage.process(data)
            
            # Log stage completion
            self.logger.log_stage_complete(
                stage.name,
                input_count=len(compounds),
                output_count=len(data)
            )
        
        return data
```

### 3.2 Resource Monitor

**Purpose**: Track computational resource usage

```python
class ResourceMonitor:
    """Monitors CPU, memory, and I/O usage"""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
    
    def sample(self):
        """Record current resource usage"""
        current_memory = get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=get_cpu_percent(),
            memory_mb=current_memory,
            io_read_mb=get_io_read(),
            io_write_mb=get_io_write()
        )
```

---

## Layer 4: Interface Layer

### 4.1 Command Line Interface

**Design**: Simple, scriptable, well-documented

```bash
# Basic usage
chemweaver -i library.smi -o results.json

# With configuration
chemweaver \
    -i library.smi \
    -o results.json \
    --confidence 0.8 \
    --uncertainty 0.3 \
    --top-n 50 \
    --seed 42 \
    --verbose
```

**Implementation**:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="ChemWeaver Virtual Screening"
    )
    
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--uncertainty", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Execute workflow
    pipeline = ScreeningPipeline(
        confidence_threshold=args.confidence,
        max_uncertainty=args.uncertainty,
        top_n=args.top_n
    )
    
    compounds = load_compounds(args.input)
    results = pipeline.screen(compounds)
    save_results(results, args.output)
```

### 4.2 Python API

**Design**: Clean, type-hinted, well-documented

```python
from chemweaver import Compound, ScreeningPipeline

# Create pipeline
pipeline = ScreeningPipeline(
    confidence_threshold=0.7,
    max_uncertainty=0.5,
    top_n=50
)

# Run screening
results = pipeline.screen(compounds)

# Process results
for result in results:
    if result.passed_filter:
        print(f"Hit: {result.compound_id}")
        print(f"  Score: {result.predicted_score:.2f}")
        print(f"  Confidence: {result.confidence:.2f}")
```

### 4.3 Container Interface

**Docker**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "chemweaver.workflow_runner"]
```

**Usage**:
```bash
docker build -t chemweaver .
docker run -v $(pwd)/data:/data chemweaver \
    -i /data/input.smi -o /data/output.json
```

---

## Data Flow

### Screening Campaign Flow

```
1. Input
   └─ Load compound library (SMILES/CSV/JSON)
           │
           ▼
2. Standardization
   └─ Validate and normalize compounds
           │
           ▼
3. Property Filtering
   └─ Apply drug-likeness filters
           │
           ▼
4. AI Scoring
   ├─ Extract features
   ├─ Predict binding affinity
   └─ Quantify uncertainty
           │
           ▼
5. Hit Selection
   └─ Select high-confidence compounds
           │
           ▼
6. Output
   ├─ Ranked results (JSON)
   ├─ Execution log
   └─ Reproducibility manifest
```

---

## Extension Points

### Adding a New Filter

```python
class CustomFilter(PropertyFilter):
    """Custom property filter"""
    
    def check(self, compound: Compound) -> bool:
        # Custom logic
        return compound.molecular_weight < 400

# Register
engine = PropertyFilterEngine()
engine.register_filter(CustomFilter())
```

### Adding a New AI Model

```python
class CustomModel(AISurrogateModel):
    """Custom AI surrogate model"""
    
    def predict_score(self, features: np.ndarray) -> float:
        # Custom prediction logic
        return self.model.predict(features)

# Use
engine = AISurrogateEngine()
engine.register_model("custom", CustomModel())
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Standardization | O(n) | O(n) |
| Property Filter | O(n) | O(1) |
| Feature Extraction | O(n × m) | O(n × f) |
| AI Inference | O(n × p) | O(n) |
| **Total** | **O(n × (m + p))** | **O(n × f)** |

Where:
- n = number of compounds
- m = average SMILES length
- p = model parameters
- f = feature dimensions

### Throughput

| Library Size | Runtime | Memory |
|--------------|---------|--------|
| 100 | <1s | <100MB |
| 1,000 | 5s | <200MB |
| 10,000 | 30s | <500MB |
| 100,000 | 5min | <2GB |
| 1,000,000 | 1hour | <8GB |

---

## Security Considerations

### Input Validation

- SMILES validation before processing
- File size limits
- Path traversal prevention
- No arbitrary code execution

### Container Security

- Non-root user execution
- Minimal base image
- No unnecessary permissions
- Regular security updates

---

## References

### Design Patterns

- **Pipeline Pattern**: Sequential data processing
- **Strategy Pattern**: Pluggable filters and models
- **Observer Pattern**: Execution monitoring
- **Factory Pattern**: Object creation

### Architectural Influences

- **Scikit-learn**: Pipeline API design
- **TensorFlow Extended (TFX)**: Production ML pipeline
- **Apache Airflow**: Workflow orchestration
- **Kubeflow**: Container-native ML

---

**Version**: 1.0.0  
**Last Updated**: 2024-02-05
