import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemweaver import Compound, MinimalScreeningPipeline
from chemweaver.ai.inference.decision_layer import UncertaintyCalibratedDecisionLayer
from chemweaver.core.inference import MinimalSurrogateModel
from chemweaver.utils.helpers import compute_reproducibility_hash


class TestDocumentationSnippets(unittest.TestCase):
    def test_readme_basic_usage_snippet_runs(self):
        compound = Compound.from_smiles(
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            compound_id="cmpd_001",
        )
        pipeline = MinimalScreeningPipeline(
            confidence_threshold=0.7,
            max_uncertainty=0.5,
            top_n=50,
        )
        results = pipeline.screen([compound])
        self.assertEqual(len(results), 1)

    def test_readme_uncertainty_snippet_runs(self):
        decision_layer = UncertaintyCalibratedDecisionLayer(
            base_threshold=0.7,
            target_difficulty="moderate",
        )
        model = MinimalSurrogateModel()
        prediction = model.predict_with_uncertainty("CCO", compound_id="cmpd_001")
        decision = decision_layer.make_decision(
            compound_id=uuid4(),
            docking_prediction=prediction.predicted_score,
            uncertainty=prediction.uncertainty,
            in_domain=True,
            domain_reliability=prediction.confidence,
        )
        self.assertIsNotNone(decision.decision)
        self.assertGreaterEqual(decision.decision_confidence, 0.0)

    def test_readme_reproducibility_hash_snippet_runs(self):
        input_file = REPO_ROOT / "data/example_compounds.smi"
        self.assertTrue(input_file.exists())
        run_hash = compute_reproducibility_hash(
            input_file=str(input_file),
            parameters={"confidence": 0.7, "uncertainty": 0.5, "top_n": 50},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.assertEqual(len(run_hash), 32)


if __name__ == "__main__":
    unittest.main()
