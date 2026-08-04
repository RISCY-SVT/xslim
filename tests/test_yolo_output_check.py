"""RISCY-SVT tests for the opt-in detector output semantic checker."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from xslim.tools.yolo_output_check import main


def identity_model(path: Path):
    value = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 6])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 6])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["images"], ["output0"])],
            "detector",
            [value],
            [output],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, path)


class TestYoloOutputCheck(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.onnx"
        identity_model(self.model)
        self.preprocess = self.root / "preprocess.py"
        self.preprocess.write_text(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "def load(path: Path):\n"
            "    return np.load(path)\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def run_case(self, value, fail=False):
        data = self.root / f"input-{len(list(self.root.glob('input-*')))}.npy"
        np.save(data, np.asarray(value, dtype=np.float32))
        image_list = self.root / "images.txt"
        image_list.write_text(f"{data}\n", encoding="utf-8")
        report = self.root / "report.json"
        arguments = [
            "--model", str(self.model),
            "--image-list", str(image_list),
            "--preprocess", f"{self.preprocess}:load",
            "--output-name", "output0",
            "--score-column", "4",
            "--class-column", "5",
            "--expected-shape", "1,3,6",
            "--report", str(report),
        ]
        if fail:
            arguments.append("--fail-on-violation")
        return main(arguments), json.loads(report.read_text(encoding="utf-8"))

    def test_noncollapsed_output_passes(self):
        value = np.zeros((1, 3, 6), dtype=np.float32)
        value[0, :, 4] = [0.1, 0.5, 0.9]
        value[0, :, 5] = [0, 1, 2]
        status, report = self.run_case(value, fail=True)
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["results"][0]["prediction_count"], 3)

    def test_collapsed_scores_are_reported_without_implicit_failure(self):
        value = np.zeros((1, 3, 6), dtype=np.float32)
        value[0, :, 5] = [0, 1, 2]
        status, report = self.run_case(value, fail=False)
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "fail")
        self.assertIn("all-zero-scores", report["violations"])

    def test_explicit_fail_closed_mode_returns_nonzero(self):
        value = np.zeros((1, 3, 6), dtype=np.float32)
        value[0, :, 5] = [0, 1, 2]
        status, _ = self.run_case(value, fail=True)
        self.assertEqual(status, 1)

    def test_nonfinite_output_is_reported(self):
        value = np.zeros((1, 3, 6), dtype=np.float32)
        value[0, :, 4] = [0.1, np.nan, 0.9]
        value[0, :, 5] = [0, 1, 2]
        _, report = self.run_case(value)
        self.assertIn("non-finite-output", report["violations"])


if __name__ == "__main__":
    unittest.main()
