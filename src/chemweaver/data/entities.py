"""
ChemWeaver Data Standard - Core Entity Definitions
=================================================

This module defines the fundamental data structures for the ChemWeaver standard,
including Compound, Target, Docking Experiment, and Screening Result entities.

Author: ChemWeaver Development Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4


class StructureSource(Enum):
    """Enumeration of supported structure sources."""
    PDB = "pdb"
    ALPHAFOLD = "alphafold"
    CRYO_EM = "cryo_em"
    HOMOLOGY_MODEL = "homology_model"
    EXPERIMENTAL = "experimental"
    PREDICTED = "predicted"


class TargetClass(Enum):
    """Target functional class ontology (simplified)."""
    ENZYME = "enzyme"
    RECEPTOR = "receptor"
    ION_CHANNEL = "ion_channel"
    TRANSPORTER = "transporter"
    TRANSCRIPTION_FACTOR = "transcription_factor"
    PROTEIN_PROTEIN_INTERFACE = "protein_protein_interface"
    UNKNOWN = "unknown"


class DockingEngine(Enum):
    """Supported docking engines."""
    AUTODOCK_GPU = "autodock_gpu"
    VINA = "vina"
    SMINA = "smina"
    GNINA = "gnina"
    GLIDE = "glide"
    GOLD = "gold"
    LEDOCK = "ledock"
    DOCK6 = "dock6"
    CUSTOM = "custom"


class ScoringFunction(Enum):
    """Supported scoring functions."""
    AUTODOCK4 = "autodock4"
    VINARDO = "vinardo"
    CNN_SCORE = "cnn_score"
    CNN_AFFINITY = "cnn_affinity"
    MM_GBSA = "mm_gbsa"
    CHEMPLP = "chemplp"
    CUSTOM = "custom"


@dataclass(frozen=True)
class MolecularIdentity:
    """
    Immutable molecular identity descriptors.
    
    All identifiers are canonicalized to ensure reproducibility across
    different chemical informatics toolkits.
    """
    canonical_smiles: str
    inchi: str
    inchikey: str
    
    # Tautomer and protonation state metadata
    tautomer_canonicalization_method: str = "rdkit_2023"
    protonation_state_ph: Optional[float] = None
    protonation_state_method: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Generate deterministic hash for molecular identity."""
        content = f"{self.canonical_smiles}:{self.inchikey}:{self.tautomer_canonicalization_method}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]


@dataclass
class ConformerMetadata:
    """Metadata for 3D conformer ensembles."""
    conformer_id: str
    generation_method: str  # e.g., "rdkit_etkdg", "obabel", "omega"
    energy: Optional[float] = None
    rmsd_to_reference: Optional[float] = None
    coordinates_blob_ref: Optional[str] = None  # Reference to external storage
    force_field: Optional[str] = None


@dataclass
class StructuralRepresentation:
    """Structural representation of a compound."""
    
    # 2D representation
    mol_block_2d: Optional[str] = None
    svg_2d: Optional[str] = None
    
    # 3D conformer ensemble
    conformers: List[ConformerMetadata] = field(default_factory=list)
    preferred_conformer_id: Optional[str] = None
    
    # Protonation state
    protonation_state: str = "neutral"
    ph: float = 7.4
    
    # Format compatibility
    compatible_formats: Set[str] = field(default_factory=lambda: {"sdf", "mol2", "pdbqt", "smi"})


@dataclass
class PhysicochemicalDescriptors:
    """Standard physicochemical descriptors following Lipinski/Veber rules."""
    molecular_weight: float
    logp: float
    tpsa: float  # Topological Polar Surface Area
    hbd: int  # Hydrogen Bond Donors
    hba: int  # Hydrogen Bond Acceptors
    rotatable_bonds: int
    heavy_atoms: int
    formal_charge: int
    
    # Extended descriptors
    mr: Optional[float] = None  # Molar Refractivity
    qed: Optional[float] = None  # Quantitative Estimate of Drug-likeness
    # ... additional descriptors can be added


@dataclass
class FingerprintDescriptor:
    """Molecular fingerprint with metadata."""
    fingerprint_type: str  # e.g., "morgan", "maccs", "ecfp4"
    fingerprint_bits: List[int]
    radius: Optional[int] = None
    n_bits: Optional[int] = None
    normalization_method: Optional[str] = None


@dataclass
class DescriptorLayer:
    """Comprehensive descriptor layer for ML compatibility."""
    physicochemical: PhysicochemicalDescriptors
    fingerprints: List[FingerprintDescriptor] = field(default_factory=list)
    
    # Feature normalization metadata
    normalization_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"molecular_weight": {"mean": 350.0, "std": 100.0}}


@dataclass
class CompoundProvenance:
    """Provenance and source tracking for compounds."""
    source_database: str
    source_version: str
    source_accession: str
    download_date: datetime
    preprocessing_pipeline_id: str
    original_format: str
    
    # Original identifiers
    original_id: Optional[str] = None
    original_smiles: Optional[str] = None


@dataclass
class SyntheticAccessibility:
    """Synthetic accessibility and developability scores."""
    sa_score: Optional[float] = None  # Synthetic accessibility (1-10, lower is easier)
    scscore: Optional[float] = None  # SCScore from Coley et al.
    # Additional metrics
    complexity_score: Optional[float] = None
    route_likelihood: Optional[float] = None


@dataclass
class ADMETPredictions:
    """ADMET prediction schema for optional model outputs."""
    caco2_permeability: Optional[float] = None
    p_glycoprotein_substrate: Optional[float] = None
    herg_inhibition: Optional[float] = None
    plasma_protein_binding: Optional[float] = None
    half_life: Optional[float] = None
    clearance: Optional[float] = None
    bioavailability: Optional[float] = None
    ld50: Optional[float] = None
    
    # Prediction metadata
    prediction_model_version: Optional[str] = None
    prediction_confidence: Optional[float] = None


@dataclass
class DrugLikenessFilters:
    """Drug-likeness filter results."""
    lipinski_violations: int = 0
    veber_violations: int = 0
    rule_of_three_pass: bool = False
    qed_score: Optional[float] = None
    bbb_permeant: Optional[bool] = None
    
    # PAINS filters
    pains_alerts: List[str] = field(default_factory=list)
    
    # Custom filters
    custom_filter_results: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Compound:
    """
    Core Compound entity in ChemWeaver.
    
    This is the fundamental unit representing a chemical compound
    with full identity, structural, descriptor, and provenance information.
    """
    
    # Core identity (required)
    identity: MolecularIdentity
    
    # Structural representations (required)
    structure: StructuralRepresentation
    
    # Descriptors (required)
    descriptors: DescriptorLayer
    
    # Provenance (required)
    provenance: CompoundProvenance
    
    # Unique identifier
    vsss_compound_id: UUID = field(default_factory=uuid4)
    
    # Developability
    synthetic_accessibility: SyntheticAccessibility = field(default_factory=lambda: SyntheticAccessibility())
    drug_likeness: DrugLikenessFilters = field(default_factory=lambda: DrugLikenessFilters())
    admet_predictions: ADMETPredictions = field(default_factory=lambda: ADMETPredictions())
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # ChemWeaver version compatibility
    schema_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize compound to dictionary."""
        return {
            "vsss_compound_id": str(self.vsss_compound_id),
            "identity": {
                "canonical_smiles": self.identity.canonical_smiles,
                "inchi": self.identity.inchi,
                "inchikey": self.identity.inchikey,
                "tautomer_canonicalization_method": self.identity.tautomer_canonicalization_method,
                "protonation_state_ph": self.identity.protonation_state_ph,
                "protonation_state_method": self.identity.protonation_state_method,
                "identity_hash": self.identity.compute_hash()
            },
            "structure": {
                "conformer_count": len(self.structure.conformers),
                "protonation_state": self.structure.protonation_state,
                "ph": self.structure.ph,
                "compatible_formats": list(self.structure.compatible_formats)
            },
            "descriptors": {
                "molecular_weight": self.descriptors.physicochemical.molecular_weight,
                "logp": self.descriptors.physicochemical.logp,
                "fingerprint_count": len(self.descriptors.fingerprints)
            },
            "provenance": {
                "source_database": self.provenance.source_database,
                "source_version": self.provenance.source_version,
                "preprocessing_pipeline_id": self.provenance.preprocessing_pipeline_id
            },
            "schema_version": self.schema_version
        }


@dataclass
class BindingSiteDefinition:
    """Binding site definition for docking experiments."""
    definition_type: str  # "grid" or "residue_list"
    
    # Grid-based definition
    grid_center: Optional[Tuple[float, float, float]] = None
    grid_dimensions: Optional[Tuple[float, float, float]] = None
    grid_spacing: Optional[float] = None
    
    # Residue-based definition
    binding_residues: List[str] = field(default_factory=list)  # e.g., ["ALA:123:A", "VAL:45:A"]
    
    # Reference ligand-based
    reference_ligand_id: Optional[str] = None
    reference_ligand_vsss_id: Optional[UUID] = None


@dataclass
class StructureConfidenceMetrics:
    """Confidence metrics for structural models."""
    overall_confidence: Optional[float] = None  # 0-100 or 0-1
    plddt: Optional[float] = None  # AlphaFold pLDDT
    local_confidence_scores: Optional[List[float]] = None
    resolution: Optional[float] = None  # For experimental structures
    r_free: Optional[float] = None
    r_work: Optional[float] = None
    clashscore: Optional[float] = None
    ramachandran_favored: Optional[float] = None


@dataclass
class Target:
    """
    Core Target entity in ChemWeaver.
    
    Represents a biological target (typically a protein) for virtual screening.
    """
    
    # Primary identifiers (required)
    uniprot_id: str
    uniprot_accession: str
    target_name: str
    
    # Unique identifier
    vsss_target_id: UUID = field(default_factory=uuid4)
    
    # Additional identifiers
    target_gene: Optional[str] = None
    organism: str = "Homo sapiens"
    
    # Functional classification
    target_class: TargetClass = TargetClass.UNKNOWN
    ec_number: Optional[str] = None  # Enzyme Commission number
    
    # Structure information
    structure_source: StructureSource = StructureSource.PDB
    structure_id: Optional[str] = None  # PDB ID or AlphaFold DB ID
    structure_file_ref: Optional[str] = None  # Path/URI to structure file
    structure_sequence: Optional[str] = None
    
    # Confidence metrics
    structure_confidence: StructureConfidenceMetrics = field(default_factory=lambda: StructureConfidenceMetrics())
    
    # Binding site
    binding_site: Optional[BindingSiteDefinition] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    schema_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize target to dictionary."""
        return {
            "vsss_target_id": str(self.vsss_target_id),
            "uniprot_id": self.uniprot_id,
            "uniprot_accession": self.uniprot_accession,
            "target_name": self.target_name,
            "target_class": self.target_class.value,
            "structure_source": self.structure_source.value,
            "structure_id": self.structure_id,
            "schema_version": self.schema_version
        }


@dataclass
class DockingParameters:
    """Docking-specific parameters."""
    exhaustiveness: int = 32
    num_modes: int = 9
    energy_range: float = 3.0
    cpu_count: int = 1
    seed: int = 42
    
    # Engine-specific parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridGenerationProtocol:
    """Grid generation protocol metadata."""
    protocol_name: str
    receptor_preparation: str  # e.g., "prepare_receptor4.py"
    grid_generation_tool: str  # e.g., "autogrid4"
    padding: float = 4.0
    spacing: float = 0.375


@dataclass
class DockingExperimentMetadata:
    """
    Comprehensive metadata for docking experiments.
    
    Captures all information necessary for full experiment reproducibility.
    """
    
    # Required fields
    docking_engine: DockingEngine
    engine_version: str
    scoring_function: ScoringFunction
    grid_protocol: GridGenerationProtocol
    execution_container_hash: str  # Docker/Singularity container digest
    workflow_definition_version: str
    parameter_template_version: str
    target_id: UUID
    compound_library_id: str
    
    # Optional/Auto fields
    experiment_id: UUID = field(default_factory=uuid4)
    experiment_name: str = "unnamed_experiment"
    engine_git_commit: Optional[str] = None
    scoring_function_version: Optional[str] = None
    parameters: DockingParameters = field(default_factory=lambda: DockingParameters())
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    execution_backend: str = "local"  # local, hpc, cloud
    schema_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "experiment_id": str(self.experiment_id),
            "experiment_name": self.experiment_name,
            "docking_engine": self.docking_engine.value,
            "engine_version": self.engine_version,
            "scoring_function": self.scoring_function.value,
            "execution_container_hash": self.execution_container_hash,
            "workflow_version": self.workflow_definition_version,
            "parameter_template_version": self.parameter_template_version,
            "execution_backend": self.execution_backend,
            "schema_version": self.schema_version
        }


@dataclass
class PoseMetadata:
    """Metadata for individual docking poses."""
    pose_id: str
    rank: int
    raw_score: float
    coordinates_blob_ref: Optional[str] = None
    rmsd_to_crystal: Optional[float] = None
    interaction_fingerprint: Optional[List[str]] = None


@dataclass
class ScreeningResult:
    """
    Core Screening Result entity in ChemWeaver.
    
    Captures the complete output of a screening operation including
    all scoring metrics and confidence estimates.
    """
    
    # Required fields
    experiment_id: UUID
    compound_id: UUID
    target_id: UUID
    raw_docking_score: float
    
    # Optional/Auto fields
    result_id: UUID = field(default_factory=uuid4)
    docking_score_unit: str = "kcal/mol"
    
    # Pose information
    poses: List[PoseMetadata] = field(default_factory=list)
    top_pose_id: Optional[str] = None
    
    # Consensus scoring
    consensus_score: Optional[float] = None
    consensus_method: Optional[str] = None
    
    # AI predictions
    ai_activity_probability: Optional[float] = None
    ai_prediction_model_id: Optional[str] = None
    ai_prediction_model_version: Optional[str] = None
    
    # Uncertainty/Confidence
    prediction_uncertainty: Optional[float] = None  # Standard deviation or confidence interval
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    prediction_domain: str = "interpolation"  # interpolation, extrapolation, out-of-domain
    
    # Clustering
    pose_cluster_id: Optional[str] = None
    cluster_representative: bool = False
    
    # Experiment context
    batch_id: str = "default_batch"
    rank_in_experiment: Optional[int] = None
    total_compounds_screened: Optional[int] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    schema_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "result_id": str(self.result_id),
            "experiment_id": str(self.experiment_id),
            "compound_id": str(self.compound_id),
            "target_id": str(self.target_id),
            "raw_docking_score": self.raw_docking_score,
            "consensus_score": self.consensus_score,
            "ai_activity_probability": self.ai_activity_probability,
            "prediction_uncertainty": self.prediction_uncertainty,
            "batch_id": self.batch_id,
            "rank_in_experiment": self.rank_in_experiment,
            "schema_version": self.schema_version
        }


# Type aliases for convenience
ChemWeaverCompound = Compound
ChemWeaverTarget = Target
ChemWeaverDockingExperiment = DockingExperimentMetadata
ChemWeaverScreeningResult = ScreeningResult
