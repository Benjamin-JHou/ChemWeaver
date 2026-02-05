"""
VS-Bench - Dataset Engineering
==============================

Reference dataset schema, metadata standards, and data management
for benchmark datasets following FAIR principles.

Author: VS-Bench Development Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4
from enum import Enum


class DataSource(Enum):
    """Supported data sources."""
    CHEMBL = "ChEMBL"
    BINDINGDB = "BindingDB"
    PDBBIND = "PDBbind"
    PUBCHEM_BIOASSAY = "PubChem BioAssay"
    IN_HOUSE = "In-House"
    ZINC = "ZINC"
    MOSES = "MOSES"


class LicenseType(Enum):
    """Dataset license types."""
    CC0 = "CC0-1.0"
    CC_BY = "CC-BY-4.0"
    CC_BY_SA = "CC-BY-SA-4.0"
    CC_BY_NC = "CC-BY-NC-4.0"
    MIT = "MIT"
    PROPRIETARY = "Proprietary"
    ACADEMIC = "Academic Use Only"


@dataclass
class MolecularIdentity:
    """Standard molecular identity information."""
    compound_id: str
    smiles: str
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute molecular hash from canonical SMILES."""
        return hashlib.sha256(self.smiles.encode()).hexdigest()[:16]


@dataclass
class ProteinIdentity:
    """Standard protein target information."""
    protein_id: str
    uniprot_id: Optional[str] = None
    protein_name: Optional[str] = None
    gene_symbol: Optional[str] = None
    protein_family: Optional[str] = None  # kinase, gpcr, protease, etc.
    organism: str = "human"
    sequence: Optional[str] = None
    pdb_id: Optional[str] = None


@dataclass
class ADMETProperties:
    """ADMET multi-task properties."""
    logd: Optional[float] = None
    solubility: Optional[float] = None  # mg/mL
    permeability: Optional[float] = None  # Papp
    hERG_inhibition: Optional[float] = None  # IC50
    CYP_inhibition: Optional[Dict[str, float]] = field(default_factory=dict)
    plasma_protein_binding: Optional[float] = None
    half_life: Optional[float] = None
    clearance: Optional[float] = None
    bioavailability: Optional[float] = None


@dataclass
class AssayMetadata:
    """Experimental assay metadata."""
    assay_id: Optional[str] = None
    assay_type: str = "binding"  # binding, functional, cellular
    standard_type: str = "IC50"  # IC50, Ki, EC50, etc.
    standard_units: str = "nM"
    activity_comment: Optional[str] = None
    data_validity: str = "valid"  # valid, suspect, invalid
    confidence_score: Optional[float] = None  # 0-1


@dataclass
class BenchmarkSample:
    """
    Single sample in benchmark dataset.
    
    Unified schema supporting multiple task types:
    - Docking surrogate prediction
    - Multi-target activity
    - ADMET properties
    """
    
    # Identifiers
    sample_id: UUID = field(default_factory=uuid4)
    
    # Molecular identity
    compound: MolecularIdentity = field(default_factory=lambda: MolecularIdentity("", ""))
    
    # Target information
    protein: Optional[ProteinIdentity] = None
    
    # Labels
    experimental_activity: Optional[float] = None  # pIC50, pKi, etc.
    docking_score: Optional[float] = None  # kcal/mol
    admet_properties: Optional[ADMETProperties] = None
    
    # Metadata
    assay_metadata: AssayMetadata = field(default_factory=AssayMetadata)
    data_source: DataSource = DataSource.CHEMBL
    collection_date: Optional[datetime] = None
    
    # Split assignment
    split: str = "train"  # train, val, test, hidden_test
    
    # Chemical features (optional, for pre-computed features)
    scaffold_id: Optional[str] = None
    murcko_scaffold: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sample_id": str(self.sample_id),
            "compound": {
                "compound_id": self.compound.compound_id,
                "smiles": self.compound.smiles,
                "inchi": self.compound.inchi,
                "inchikey": self.compound.inchikey,
                "hash": self.compound.compute_hash()
            },
            "protein": {
                "protein_id": self.protein.protein_id if self.protein else None,
                "uniprot_id": self.protein.uniprot_id if self.protein else None,
                "family": self.protein.protein_family if self.protein else None
            } if self.protein else None,
            "labels": {
                "experimental_activity": self.experimental_activity,
                "docking_score": self.docking_score,
                "admet": self._admet_to_dict() if self.admet_properties else None
            },
            "metadata": {
                "assay": self.assay_metadata.__dict__ if self.assay_metadata else None,
                "source": self.data_source.value,
                "collection_date": self.collection_date.isoformat() if self.collection_date else None
            },
            "split": self.split,
            "scaffold": {
                "scaffold_id": self.scaffold_id,
                "murcko": self.murcko_scaffold
            }
        }
    
    def _admet_to_dict(self) -> Dict[str, Any]:
        """Convert ADMET properties to dict."""
        if not self.admet_properties:
            return {}
        return {
            k: v for k, v in self.admet_properties.__dict__.items()
            if v is not None
        }


@dataclass
class DatasetDataCard:
    """
    Dataset metadata card following FAIR principles.
    
    Provides comprehensive documentation of dataset
    characteristics, biases, and usage guidelines.
    """
    
    # Identification
    dataset_id: UUID = field(default_factory=uuid4)
    dataset_name: str = ""
    dataset_version: str = "1.0.0"
    
    # Description
    description: str = ""
    long_description: str = ""
    intended_use: str = ""
    
    # Data sources
    primary_sources: List[DataSource] = field(default_factory=list)
    source_versions: Dict[str, str] = field(default_factory=dict)
    
    # Collection
    collection_methodology: str = ""
    curation_pipeline: str = ""
    date_created: datetime = field(default_factory=datetime.utcnow)
    date_modified: datetime = field(default_factory=datetime.utcnow)
    
    # Statistics
    total_samples: int = 0
    num_compounds: int = 0
    num_targets: int = 0
    num_scaffolds: int = 0
    
    # Splits
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    split_strategy: str = ""
    
    # Quality
    data_quality_score: Optional[float] = None  # 0-1
    known_biases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Legal
    license: LicenseType = LicenseType.CC_BY
    citation: str = ""
    doi: Optional[str] = None
    
    # Maintenance
    maintainer: str = ""
    contact_email: str = ""
    update_frequency: str = "annual"  # annual, quarterly, continuous
    
    def to_markdown(self) -> str:
        """Generate markdown data card."""
        lines = [
            f"# {self.dataset_name}",
            f"**Version**: {self.dataset_version}",
            f"**Dataset ID**: {self.dataset_id}",
            "",
            "## Description",
            self.description,
            "",
            "## Intended Use",
            self.intended_use,
            "",
            "## Data Sources",
        ]
        
        for source in self.primary_sources:
            version = self.source_versions.get(source.value, "unknown")
            lines.append(f"- {source.value} (version: {version})")
        
        lines.extend([
            "",
            "## Statistics",
            f"- Total samples: {self.total_samples:,}",
            f"- Unique compounds: {self.num_compounds:,}",
            f"- Protein targets: {self.num_targets:,}",
            f"- Unique scaffolds: {self.num_scaffolds:,}",
            "",
            "## Splits",
            f"- Training: {self.train_size:,}",
            f"- Validation: {self.val_size:,}",
            f"- Test: {self.test_size:,}",
            f"- Split strategy: {self.split_strategy}",
            "",
            "## Known Biases",
        ])
        
        for bias in self.known_biases:
            lines.append(f"- {bias}")
        
        lines.extend([
            "",
            "## Limitations",
        ])
        
        for limitation in self.limitations:
            lines.append(f"- {limitation}")
        
        lines.extend([
            "",
            "## License",
            f"{self.license.value}",
            "",
            "## Citation",
            self.citation,
            "",
            f"**DOI**: {self.doi if self.doi else 'Pending'}"
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": str(self.dataset_id),
            "name": self.dataset_name,
            "version": self.dataset_version,
            "description": self.description,
            "sources": [s.value for s in self.primary_sources],
            "statistics": {
                "total_samples": self.total_samples,
                "num_compounds": self.num_compounds,
                "num_targets": self.num_targets,
                "num_scaffolds": self.num_scaffolds
            },
            "splits": {
                "train": self.train_size,
                "val": self.val_size,
                "test": self.test_size,
                "strategy": self.split_strategy
            },
            "quality": {
                "score": self.data_quality_score,
                "biases": self.known_biases,
                "limitations": self.limitations
            },
            "license": self.license.value,
            "doi": self.doi,
            "maintainer": self.maintainer,
            "created": self.date_created.isoformat()
        }


class BenchmarkDataset:
    """
    Complete benchmark dataset with schema enforcement.
    
    Manages dataset samples, splits, and metadata following
    FAIR principles and benchmark requirements.
    """
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        data_card: Optional[DatasetDataCard] = None
    ):
        self.dataset_id = uuid4()
        self.name = name
        self.version = version
        self.data_card = data_card or DatasetDataCard(
            dataset_name=name,
            dataset_version=version
        )
        
        # Storage
        self.samples: Dict[UUID, BenchmarkSample] = {}
        self.compound_index: Dict[str, Set[UUID]] = {}  # compound_id -> sample_ids
        self.protein_index: Dict[str, Set[UUID]] = {}   # protein_id -> sample_ids
        self.scaffold_index: Dict[str, Set[UUID]] = {}  # scaffold_id -> sample_ids
        
        # Split tracking
        self.splits: Dict[str, List[UUID]] = {
            "train": [],
            "val": [],
            "test": [],
            "hidden_test": []
        }
    
    def add_sample(self, sample: BenchmarkSample) -> UUID:
        """Add a sample to the dataset."""
        self.samples[sample.sample_id] = sample
        
        # Update indices
        compound_id = sample.compound.compound_id
        if compound_id not in self.compound_index:
            self.compound_index[compound_id] = set()
        self.compound_index[compound_id].add(sample.sample_id)
        
        if sample.protein:
            protein_id = sample.protein.protein_id
            if protein_id not in self.protein_index:
                self.protein_index[protein_id] = set()
            self.protein_index[protein_id].add(sample.sample_id)
        
        if sample.scaffold_id:
            if sample.scaffold_id not in self.scaffold_index:
                self.scaffold_index[sample.scaffold_id] = set()
            self.scaffold_index[sample.scaffold_id].add(sample.sample_id)
        
        # Update split
        if sample.split in self.splits:
            self.splits[sample.split].append(sample.sample_id)
        
        return sample.sample_id
    
    def get_split(self, split_name: str) -> List[BenchmarkSample]:
        """Get all samples in a split."""
        sample_ids = self.splits.get(split_name, [])
        return [self.samples[sid] for sid in sample_ids if sid in self.samples]
    
    def get_by_compound(self, compound_id: str) -> List[BenchmarkSample]:
        """Get all samples for a compound."""
        sample_ids = self.compound_index.get(compound_id, set())
        return [self.samples[sid] for sid in sample_ids]
    
    def get_by_protein(self, protein_id: str) -> List[BenchmarkSample]:
        """Get all samples for a protein target."""
        sample_ids = self.protein_index.get(protein_id, set())
        return [self.samples[sid] for sid in sample_ids]
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        # Update data card
        self.data_card.total_samples = len(self.samples)
        self.data_card.num_compounds = len(self.compound_index)
        self.data_card.num_targets = len(self.protein_index)
        self.data_card.num_scaffolds = len(self.scaffold_index)
        self.data_card.train_size = len(self.splits["train"])
        self.data_card.val_size = len(self.splits["val"])
        self.data_card.test_size = len(self.splits["test"])
        
        # Compute label statistics
        activities = [
            s.experimental_activity for s in self.samples.values()
            if s.experimental_activity is not None
        ]
        
        docking_scores = [
            s.docking_score for s in self.samples.values()
            if s.docking_score is not None
        ]
        
        return {
            "samples": len(self.samples),
            "compounds": len(self.compound_index),
            "targets": len(self.protein_index),
            "scaffolds": len(self.scaffold_index),
            "splits": {
                k: len(v) for k, v in self.splits.items()
            },
            "activity": {
                "count": len(activities),
                "mean": sum(activities) / len(activities) if activities else None,
                "min": min(activities) if activities else None,
                "max": max(activities) if activities else None
            },
            "docking": {
                "count": len(docking_scores),
                "mean": sum(docking_scores) / len(docking_scores) if docking_scores else None,
                "min": min(docking_scores) if docking_scores else None,
                "max": max(docking_scores) if docking_scores else None
            }
        }
    
    def export_to_parquet(
        self,
        output_dir: str,
        splits: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Export dataset to Parquet format."""
        import pandas as pd
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        splits_to_export = splits or ["train", "val", "test"]
        
        for split in splits_to_export:
            samples = self.get_split(split)
            if not samples:
                continue
            
            # Convert to DataFrame
            data = [s.to_dict() for s in samples]
            df = pd.json_normalize(data)
            
            # Save
            file_path = output_path / f"{split}.parquet"
            df.to_parquet(file_path, index=False)
            exported[split] = str(file_path)
        
        # Save metadata
        metadata_path = output_path / "metadata.json"
        metadata = {
            "dataset_id": str(self.dataset_id),
            "name": self.name,
            "version": self.version,
            "statistics": self.compute_statistics(),
            "data_card": self.data_card.to_dict()
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        exported["metadata"] = str(metadata_path)
        
        return exported
    
    def validate_schema(self) -> Tuple[bool, List[str]]:
        """Validate dataset against schema requirements."""
        errors = []
        
        # Check required fields
        for sample_id, sample in self.samples.items():
            if not sample.compound.smiles:
                errors.append(f"Sample {sample_id}: Missing SMILES")
            
            if sample.experimental_activity is None and sample.docking_score is None:
                errors.append(f"Sample {sample_id}: No label provided")
        
        # Check split balance
        total = sum(len(v) for v in self.splits.values())
        if total != len(self.samples):
            errors.append(f"Split assignments don't match total samples")
        
        return len(errors) == 0, errors


class DatasetBuilder:
    """
    Builder for creating benchmark datasets.
    
    Supports data from multiple sources with proper
    validation, splitting, and metadata generation.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.dataset = BenchmarkDataset(name, version)
        self.data_sources: List[DataSource] = []
    
    def add_chembl_data(
        self,
        assay_id: str,
        standard_type: str = "IC50",
        min_confidence: int = 8
    ) -> DatasetBuilder:
        """Add data from ChEMBL (placeholder for actual implementation)."""
        self.data_sources.append(DataSource.CHEMBL)
        # Actual implementation would query ChEMBL API
        return self
    
    def add_bindingdb_data(
        self,
        protein_target: Optional[str] = None
    ) -> DatasetBuilder:
        """Add data from BindingDB."""
        self.data_sources.append(DataSource.BINDINGDB)
        return self
    
    def add_pdbbind_data(
        self,
        year: int = 2020,
        include_general_set: bool = True
    ) -> DatasetBuilder:
        """Add data from PDBbind."""
        self.data_sources.append(DataSource.PDBBIND)
        return self
    
    def apply_scaffold_split(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        random_seed: int = 42
    ) -> DatasetBuilder:
        """Apply scaffold-based split to dataset."""
        # Implementation would group by scaffold and split
        self.dataset.data_card.split_strategy = "scaffold"
        return self
    
    def apply_temporal_split(
        self,
        train_before: str = "2020-01-01",
        val_before: str = "2021-01-01"
    ) -> DatasetBuilder:
        """Apply temporal split to dataset."""
        self.dataset.data_card.split_strategy = "temporal"
        return self
    
    def set_license(self, license: LicenseType) -> DatasetBuilder:
        """Set dataset license."""
        self.dataset.data_card.license = license
        return self
    
    def add_bias_warning(self, bias_description: str) -> DatasetBuilder:
        """Add known bias warning."""
        self.dataset.data_card.known_biases.append(bias_description)
        return self
    
    def build(self) -> BenchmarkDataset:
        """Build and validate the dataset."""
        self.dataset.data_card.primary_sources = list(set(self.data_sources))
        
        # Validate
        is_valid, errors = self.dataset.validate_schema()
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {errors}")
        
        # Compute final statistics
        self.dataset.compute_statistics()
        
        return self.dataset


# Export
__all__ = [
    'DataSource',
    'LicenseType',
    'MolecularIdentity',
    'ProteinIdentity',
    'ADMETProperties',
    'AssayMetadata',
    'BenchmarkSample',
    'DatasetDataCard',
    'BenchmarkDataset',
    'DatasetBuilder'
]
