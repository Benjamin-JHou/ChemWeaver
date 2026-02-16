import json
import tempfile
import unittest
from pathlib import Path
import sys


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemweaver.workflow_runner import WorkflowRunner


class TestWorkflowRunner(unittest.TestCase):
    def test_workflow_runner_end_to_end_with_smi_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "compounds.smi"
            input_path.write_text("CCO cmpd_a\nCCN cmpd_b\n")
            output_path = tmp_path / "results.json"

            runner = WorkflowRunner()
            summary = runner.run(
                input_path=str(input_path),
                output_path=str(output_path),
                confidence_threshold=0.0,
                max_uncertainty=1.0,
                top_n=1,
            )

            self.assertEqual(summary["status"], "SUCCESS")
            self.assertEqual(summary["total_compounds"], 2)
            self.assertEqual(summary["compounds_processed"], 2)
            self.assertTrue(output_path.exists())

            payload = json.loads(output_path.read_text())
            self.assertEqual(payload["total_compounds"], 2)
            self.assertEqual(len(payload["results"]), 2)
            self.assertEqual(payload["selected_hits"], 2)


if __name__ == "__main__":
    unittest.main()
