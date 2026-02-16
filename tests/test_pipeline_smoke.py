import json
import tempfile
import unittest
from pathlib import Path
import sys


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemweaver.core.pipeline import Compound, MinimalScreeningPipeline


class TestPipelineSmoke(unittest.TestCase):
    def test_pipeline_screen_select_and_save(self):
        compounds = [
            Compound.from_smiles("CCO", "cmpd_1"),
            Compound.from_smiles("CCN", "cmpd_2"),
            Compound.from_smiles("CCCO", "cmpd_3"),
        ]

        pipeline = MinimalScreeningPipeline(
            confidence_threshold=0.0,
            max_uncertainty=1.0,
            top_n=2,
        )
        results = pipeline.screen(compounds)

        self.assertEqual(len(results), 3)
        scores = [result.predicted_score for result in results]
        self.assertEqual(scores, sorted(scores))

        hits = pipeline.select_hits(results)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(hit.passed_filter for hit in hits))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "results.json"
            pipeline.save_results(results, str(output_path))
            payload = json.loads(output_path.read_text())

        self.assertEqual(payload["total_compounds"], 3)
        self.assertEqual(payload["selected_hits"], 3)
        self.assertEqual(len(payload["results"]), 3)
        self.assertEqual(payload["parameters"]["top_n"], 2)


if __name__ == "__main__":
    unittest.main()
