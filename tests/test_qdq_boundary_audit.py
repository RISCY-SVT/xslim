"""RISCY-SVT synthetic saturation tests for the Q/DQ boundary audit."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from xslim.tools.qdq_boundary_audit import main


def float_model(path: Path, shape):
    value = helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("boundary", TensorProto.FLOAT, shape)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["images"], ["boundary"])],
            "float-boundary",
            [value],
            [output],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, path)


def quant_model(path: Path, shape, scale, zero_point, axis=None):
    value = helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("boundary", TensorProto.FLOAT, shape)
    attributes = {} if axis is None else {"axis": axis}
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantizeLinear",
                ["images", "scale", "zero_point"],
                ["boundary_q"],
                **attributes,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["boundary_q", "scale", "zero_point"],
                ["boundary"],
                **attributes,
            ),
        ],
        "quant-boundary",
        [value],
        [output],
        [
            numpy_helper.from_array(np.asarray(scale, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.asarray(zero_point, dtype=np.int8), "zero_point"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


class TestQdqBoundaryAudit(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.preprocess = self.root / "preprocess.py"
        self.preprocess.write_text(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "def load(path: Path):\n"
            "    return np.load(path)\n",
            encoding="utf-8",
        )
        self.tensors = self.root / "tensors.txt"
        self.tensors.write_text("boundary\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def audit(self, value, scale, zero_point, axis=None):
        shape = list(value.shape)
        float_path = self.root / "float.onnx"
        quant_path = self.root / "quant.onnx"
        float_model(float_path, shape)
        quant_model(quant_path, shape, scale, zero_point, axis)
        data = self.root / "input.npy"
        np.save(data, value.astype(np.float32))
        image_list = self.root / "images.txt"
        image_list.write_text(f"{data}\n", encoding="utf-8")
        report = self.root / "report.tsv"
        status = main([
            "--float-model", str(float_path),
            "--quant-model", str(quant_path),
            "--tensor-list", str(self.tensors),
            "--image-list", str(image_list),
            "--preprocess", f"{self.preprocess}:load",
            "--report", str(report),
        ])
        with report.open(encoding="utf-8", newline="") as source:
            row = next(csv.DictReader(source, delimiter="\t"))
        return status, row

    def test_per_tensor_saturation_is_counted(self):
        value = np.asarray([[-100.0, -1.0, 1.0, 100.0]], dtype=np.float32)
        status, row = self.audit(value, 0.5, -2)
        self.assertEqual(status, 0)
        self.assertEqual(row["dtype"], "int8")
        self.assertEqual(row["granularity"], "per-tensor")
        self.assertEqual(int(row["below_range_count"]), 1)
        self.assertEqual(int(row["above_range_count"]), 1)
        self.assertEqual(int(row["quantized_min_rail_hits"]), 1)
        self.assertEqual(int(row["quantized_max_rail_hits"]), 1)
        self.assertEqual(int(row["dequantized_min_rail_hits"]), 1)
        self.assertEqual(int(row["dequantized_max_rail_hits"]), 1)

    def test_per_channel_axis_is_reported(self):
        value = np.asarray([[[-100.0, 1.0], [2.0, 100.0]]], dtype=np.float32)
        status, row = self.audit(value, [0.5, 1.0], [-2, 3], axis=1)
        self.assertEqual(status, 0)
        self.assertEqual(row["granularity"], "per-channel")
        self.assertEqual(row["axis"], "1")
        self.assertGreaterEqual(int(row["quantized_min_rail_hits"]), 1)


if __name__ == "__main__":
    unittest.main()
