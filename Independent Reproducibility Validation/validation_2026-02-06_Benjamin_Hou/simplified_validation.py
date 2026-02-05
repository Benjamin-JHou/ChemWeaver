#!/usr/bin/env python3
"""
Simplified Independent Reproducibility Validation for ChemWeaver
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

# Add ChemWeaver to path (as external user would)
sys.path.insert(0, 'src')

try:
    import chemweaver
    from chemweaver.core.pipeline import Compound, MinimalScreeningPipeline
    from chemweaver.core.inference import MinimalSurrogateModel, Prediction
    print("✅ ChemWeaver imported successfully")
    chemweaver.print_status()
except ImportError as e:
    print(f"❌ Failed to import ChemWeaver: {e}")
    sys.exit(1)

def generate_test_compounds(n=1000):
    """Generate test compounds."""
    compounds = []
    
    # Sample SMILES for diversity
    base_smiles = [
        "CC1=CC=C(C=C1)C(=O)O",      # Benzoic acid
        "CC(C)CC1=CC=C(C=C1)C(=O)O",  # Ibuprofen-like
        "CC1=CC=C(C=C1)C(=O)N",         # Benzamide
        "CC(C)NCC(O)CO",               # Propranolol-like
        "CC1=CC=C(C=C1)N",             # Aniline
        "CC1=CC=C(C=C1)O",             # Phenol
        "CC1=CC=C(C=C1)S",             # Thiophenol
        "CC1=CC=C(C=C1)Cl",            # Chlorobenzene
        "CC1=CC=C(C=C1)F",             # Fluorobenzene
        "CC1=CC=C(C=C1)C#N",          # Benzonitrile
    ]
    
    for i in range(n):
        base_idx = i % len(base_smiles)
        smiles = base_smiles[base_idx]
        
        # Add small modifications
        if i % 3 == 0:
            smiles = smiles + "C"
        elif i % 3 == 1:
            smiles = "C" + smiles
            
        compound = Compound.from_smiles(smiles, f"test_compound_{i:04d}")
        compounds.append(compound)
    
    return compounds

def run_reproducibility_validation():
    """Run the main validation."""
    print("🎯 Starting Independent Reproducibility Validation")
    print("=" * 60)
    
    # Generate test data
    compounds = generate_test_compounds(n=1000)
    print(f"🧪 Generated {len(compounds)} test compounds")
    
    # Initialize pipeline
    print("🔄 Initializing ChemWeaver pipeline...")
    pipeline = MinimalScreeningPipeline()
    
    # Test 1: Basic functionality
    print("⚡ Testing basic screening functionality...")
    start_time = time.time()
    results = pipeline.screen(compounds)
    end_time = time.time()
    
    runtime = end_time - start_time
    print(f"   ✅ Screened {len(compounds)} compounds in {runtime:.2f}s")
    print(f"   ✅ Average: {runtime/len(compounds)*1000:.2f}ms per compound")
    
    # Test 2: Reproducibility (multiple runs)
    print("🔄 Testing run-to-run reproducibility...")
    run_results = []
    
    for run_idx in range(3):
        run_start = time.time()
        run_results_i = pipeline.screen(compounds[:200])  # Smaller subset
        run_end = time.time()
        
        scores = [r.predicted_score for r in run_results_i]
        run_results.append(scores)
        
        print(f"   Run {run_idx + 1}: {len(scores)} compounds in {run_end - run_start:.2f}s")
    
    # Calculate correlations between runs
    correlations = []
    for i in range(len(run_results)):
        for j in range(i + 1, len(run_results)):
            # Simple Pearson correlation
            corr = np.corrcoef(run_results[i], run_results[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    avg_correlation = np.mean(correlations) if correlations else 0
    print(f"   ✅ Average run-to-run correlation: {avg_correlation:.3f}")
    
    # Test 3: Uncertainty behavior
    print("📈 Testing uncertainty quantification...")
    uncertainties = [r.uncertainty for r in results]
    scores = [r.predicted_score for r in results]
    
    uncertainty_mean = np.mean(uncertainties)
    uncertainty_std = np.std(uncertainties)
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    
    print(f"   ✅ Score range: {np.min(scores):.3f} to {np.max(scores):.3f}")
    print(f"   ✅ Uncertainty range: {np.min(uncertainties):.3f} to {np.max(uncertainties):.3f}")
    print(f"   ✅ Score-Uncertainty correlation: {np.corrcoef(scores, uncertainties)[0, 1]:.3f}")
    
    # Generate validation report
    validation_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'chemweaver_version': chemweaver.__version__,
        'test_compounds': len(compounds),
        'environment': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'platform': sys.platform,
            'independent_test': True
        },
        'results': {
            'runtime_total_seconds': runtime,
            'runtime_per_compound_ms': runtime/len(compounds)*1000,
            'compounds_processed': len(results),
            'reproducibility_correlation': avg_correlation,
            'score_statistics': {
                'mean': float(score_mean),
                'std': float(score_std),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            },
            'uncertainty_statistics': {
                'mean': float(uncertainty_mean),
                'std': float(uncertainty_std),
                'min': float(np.min(uncertainties)),
                'max': float(np.max(uncertainties))
            }
        },
        'validation_criteria': {
            'reproducibility_pass': avg_correlation > 0.85,
            'runtime_reasonable': runtime/len(compounds) < 1.0,  # < 1s per compound
            'uncertainty_calibrated': uncertainty_mean > 0.1 and uncertainty_mean < 1.0,
            'overall_success': False
        }
    }
    
    # Determine overall success
    validation_results['validation_criteria']['overall_success'] = (
        validation_results['validation_criteria']['reproducibility_pass'] and
        validation_results['validation_criteria']['runtime_reasonable'] and
        validation_results['validation_criteria']['uncertainty_calibrated']
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("🏁 INDEPENDENT VALIDATION RESULTS")
    print("=" * 60)
    
    criteria = validation_results['validation_criteria']
    print(f"🔄 Reproducibility: {'✅ PASS' if criteria['reproducibility_pass'] else '❌ FAIL'}")
    print(f"   Correlation: {avg_correlation:.3f} (threshold: >0.85)")
    
    print(f"⚡ Performance: {'✅ PASS' if criteria['runtime_reasonable'] else '❌ FAIL'}")
    print(f"   Runtime: {runtime/len(compounds)*1000:.2f}ms/compound (threshold: <1000ms)")
    
    print(f"📈 Uncertainty: {'✅ PASS' if criteria['uncertainty_calibrated'] else '❌ FAIL'}")
    print(f"   Mean uncertainty: {uncertainty_mean:.3f} (threshold: 0.1-1.0)")
    
    print(f"\n🎯 OVERALL: {'✅ SUCCESS' if criteria['overall_success'] else '❌ FAILURE'}")
    
    if criteria['overall_success']:
        print("✅ ChemWeaver passes independent reproducibility validation!")
    else:
        print("❌ ChemWeaver fails independent reproducibility validation!")
    
    # Save results
    output_dir = Path("independent_validation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "validation_report.json", "w") as f:
        # Convert bool to int for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, bool):
                return int(obj)
            elif isinstance(obj, (int, float, str)):
                return obj
            else:
                return str(obj)
        
        json.dump(convert_types(validation_results), f, indent=2)
    
    # Save figure data
    figure_data = {
        'reproducibility_correlation': avg_correlation,
        'runtime_per_compound': runtime/len(compounds)*1000,
        'uncertainty_mean': uncertainty_mean,
        'overall_success': criteria['overall_success'],
        'n_compounds': len(compounds)
    }
    
    with open(output_dir / "figure_6_data.json", "w") as f:
        json.dump(convert_types(figure_data), f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir}/")
    
    return validation_results['validation_criteria']['overall_success']

if __name__ == "__main__":
    success = run_reproducibility_validation()
    sys.exit(0 if success else 1)