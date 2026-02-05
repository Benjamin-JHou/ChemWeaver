"""
VSSS Workflow Runner
====================

Command-line interface and workflow orchestration for the
minimal VSSS screening pipeline.

Provides:
- CLI for running screening workflows
- Configuration management
- Logging and monitoring
- Result reporting

Usage:
    python workflow_runner.py --input compounds.smi --output results.json
    python workflow_runner.py --example

Author: VSSS Development Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline import Compound, MinimalScreeningPipeline


class WorkflowRunner:
    """
    Workflow runner for VSSS screening pipelines.
    
    Manages:
    - Input/output handling
    - Configuration management
    - Execution monitoring
    - Logging
    
    Example:
        >>> runner = WorkflowRunner(config)
        >>> results = runner.run(input_file, output_file)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize workflow runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.execution_log = []
        self.start_time = None
        self.end_time = None
    
    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log execution message.
        
        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.execution_log.append(log_entry)
        
        # Also print to console
        print(f"[{level}] {message}")
    
    def load_compounds(self, input_path: str) -> List[Compound]:
        """
        Load compounds from input file.
        
        Supports:
        - .smi: SMILES strings (one per line)
        - .csv: CSV with 'smiles' column
        - .json: JSON array of compounds
        
        Args:
            input_path: Path to input file
            
        Returns:
            List of Compound objects
        """
        self.log(f"Loading compounds from: {input_path}")
        
        path = Path(input_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        compounds = []
        
        if path.suffix == '.smi':
            # SMILES format
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Check if line has ID
                        parts = line.split()
                        if len(parts) >= 2:
                            smiles, cid = parts[0], parts[1]
                        else:
                            smiles = parts[0]
                            cid = f"cmpd_{i:05d}"
                        
                        compound = Compound.from_smiles(smiles, compound_id=cid)
                        compounds.append(compound)
        
        elif path.suffix == '.csv':
            # CSV format
            import csv
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    smiles = row.get('smiles', row.get('SMILES', ''))
                    cid = row.get('id', row.get('ID', f"cmpd_{i:05d}"))
                    
                    if smiles:
                        compound = Compound.from_smiles(smiles, compound_id=cid)
                        compounds.append(compound)
        
        elif path.suffix == '.json':
            # JSON format
            with open(path, 'r') as f:
                data = json.load(f)
                
            for item in data:
                smiles = item.get('smiles', item.get('SMILES', ''))
                cid = item.get('id', item.get('compound_id', ''))
                
                if smiles:
                    compound = Compound.from_smiles(smiles, compound_id=cid)
                    compounds.append(compound)
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self.log(f"Loaded {len(compounds)} compounds")
        return compounds
    
    def run(
        self,
        input_path: str,
        output_path: str,
        confidence_threshold: float = 0.7,
        max_uncertainty: float = 0.5,
        top_n: int = 50
    ) -> Dict[str, Any]:
        """
        Run complete screening workflow.
        
        Args:
            input_path: Path to input compound file
            output_path: Path to output results file
            confidence_threshold: Minimum confidence for hits
            max_uncertainty: Maximum uncertainty for hits
            top_n: Number of top hits to return
            
        Returns:
            Execution summary dictionary
        """
        self.start_time = datetime.utcnow()
        self.log("Starting VSSS Screening Workflow")
        self.log(f"Configuration: confidence={confidence_threshold}, "
                f"uncertainty={max_uncertainty}, top_n={top_n}")
        
        try:
            # Load compounds
            compounds = self.load_compounds(input_path)
            
            # Initialize pipeline
            pipeline = MinimalScreeningPipeline(
                confidence_threshold=confidence_threshold,
                max_uncertainty=max_uncertainty,
                top_n=top_n
            )
            
            # Run screening
            self.log("Running screening pipeline...")
            results = pipeline.screen(compounds)
            
            # Select hits
            hits = pipeline.select_hits(results)
            
            # Save results
            pipeline.save_results(results, output_path)
            
            # Generate summary
            self.end_time = datetime.utcnow()
            duration = (self.end_time - self.start_time).total_seconds()
            
            summary = {
                'status': 'SUCCESS',
                'input_file': input_path,
                'output_file': output_path,
                'total_compounds': len(compounds),
                'compounds_processed': len(results),
                'hits_selected': len(hits),
                'duration_seconds': duration,
                'pipeline_id': pipeline.pipeline_id,
                'timestamp': self.start_time.isoformat()
            }
            
            self.log(f"Workflow completed successfully in {duration:.2f} seconds")
            self.log(f"Selected {len(hits)} hits from {len(compounds)} compounds")
            
            return summary
            
        except Exception as e:
            self.end_time = datetime.utcnow()
            self.log(f"Workflow failed: {str(e)}", level="ERROR")
            
            return {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': self.start_time.isoformat()
            }
    
    def save_execution_log(self, log_path: str) -> None:
        """
        Save execution log to file.
        
        Args:
            log_path: Path to log file
        """
        with open(log_path, 'w') as f:
            json.dump(self.execution_log, f, indent=2)
        
        print(f"Execution log saved to: {log_path}")


def create_example_data(output_path: str = "example_compounds.smi") -> str:
    """
    Create example compound library for testing.
    
    Args:
        output_path: Path to save example data
        
    Returns:
        Path to created file
    """
    example_compounds = """# Example compound library for VSSS screening
# Format: SMILES ID
CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen_like_001
c1ccc(cc1)C(=O)O benzoic_acid_001
CC(C)NCC(COc1ccccc1)O propranolol_like_001
Cc1ccc(cc1)S(=O)(=O)N tosylamide_001
CC(=O)Nc1ccc(cc1)O paracetamol_like_001
CCOc1ccccc1 phenetole_001
CC(C)C(C)C(=O)O valeric_acid_001
CC(C)CC(C)C isooctane_001
"""
    
    with open(output_path, 'w') as f:
        f.write(example_compounds)
    
    print(f"Example data created: {output_path}")
    return output_path


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="VSSS Virtual Screening Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with example data
  python workflow_runner.py --example
  
  # Run with custom input
  python workflow_runner.py -i compounds.smi -o results.json
  
  # Adjust selection criteria
  python workflow_runner.py -i compounds.smi -o results.json \\
      --confidence 0.8 --uncertainty 0.3 --top-n 100
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input compound file (.smi, .csv, .json)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='screening_results.json',
        help='Output results file (default: screening_results.json)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Minimum confidence threshold (default: 0.7)'
    )
    
    parser.add_argument(
        '--uncertainty',
        type=float,
        default=0.5,
        help='Maximum uncertainty threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='Number of top hits to return (default: 50)'
    )
    
    parser.add_argument(
        '--example',
        action='store_true',
        help='Run with example data'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='VSSS Workflow Runner 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Handle example mode
    if args.example:
        print("="*60)
        print("VSSS Workflow Runner - Example Mode")
        print("="*60)
        input_file = create_example_data()
    elif args.input:
        input_file = args.input
    else:
        parser.print_help()
        sys.exit(1)
    
    # Run workflow
    runner = WorkflowRunner()
    summary = runner.run(
        input_path=input_file,
        output_path=args.output,
        confidence_threshold=args.confidence,
        max_uncertainty=args.uncertainty,
        top_n=args.top_n
    )
    
    # Save execution log
    log_path = args.output.replace('.json', '_log.json')
    runner.save_execution_log(log_path)
    
    # Print summary
    print("\n" + "="*60)
    print("WORKFLOW SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['status'] == 'SUCCESS' else 1)


if __name__ == "__main__":
    main()
