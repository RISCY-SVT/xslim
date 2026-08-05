"""RISCY-SVT tests for the sanitized split workflow example."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "k1x_yolo26_split"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestK1xYolo26SplitExample(unittest.TestCase):
    def _config(self, name: str) -> dict:
        return json.loads((SAMPLE / name).read_text())

    def test_configs_preserve_six_identical_separate_vendor_boundaries(self):
        repro = self._config("config_stage64_repro.json")
        accuracy = self._config("config_accuracy_starting_point.json")
        repro_boundaries = repro["quantization_parameters"]["truncate_var_names"]
        accuracy_boundaries = accuracy["quantization_parameters"]["truncate_var_names"]
        self.assertEqual(len(repro_boundaries), 6)
        self.assertEqual(len(set(repro_boundaries)), 6)
        self.assertEqual(repro_boundaries, accuracy_boundaries)

    def test_stage64_reproduction_policy_is_exact(self):
        config = self._config("config_stage64_repro.json")
        calibration = config["calibration_parameters"]
        input_parameter = calibration["input_parameters"][0]
        quantization = config["quantization_parameters"]
        self.assertEqual(calibration["calibration_step"], 50)
        self.assertEqual(input_parameter["std_value"], [1, 1, 1])
        self.assertEqual(
            input_parameter["preprocess_file"], "./preprocess.py:preprocess_impl"
        )
        self.assertEqual(quantization["precision_level"], 0)
        self.assertEqual(quantization["finetune_level"], 1)
        self.assertTrue(quantization["analysis_enable"])

    def test_accuracy_starting_point_is_distinct_and_unvalidated(self):
        config = self._config("config_accuracy_starting_point.json")
        calibration = config["calibration_parameters"]
        input_parameter = calibration["input_parameters"][0]
        quantization = config["quantization_parameters"]
        self.assertEqual(calibration["calibration_step"], 500)
        self.assertIn("500plus", input_parameter["data_list_path"])
        self.assertEqual(input_parameter["std_value"], [1, 1, 1])
        self.assertEqual(quantization["precision_level"], 1)
        self.assertEqual(quantization["finetune_level"], 2)
        readme = (SAMPLE / "README.md").read_text()
        self.assertIn("has not passed K1X board or COCO validation", readme)

    def test_configs_are_public_safe_placeholders(self):
        self.assertFalse((SAMPLE / "config.json").exists())
        for name in (
            "config_stage64_repro.json",
            "config_accuracy_starting_point.json",
        ):
            text = (SAMPLE / name).read_text()
            self.assertNotIn("/data/", text)
            self.assertNotIn(".onnxruntime", text)

    def test_preprocess_is_nchw_rgb_float_letterbox(self):
        preprocess = _module("riscy_sample_preprocess", SAMPLE / "preprocess.py")
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            bgr = np.zeros((320, 640, 3), dtype=np.uint8)
            bgr[:, :, 2] = 255
            self.assertTrue(cv2.imwrite(str(image_path), bgr))
            value = preprocess.preprocess_one(image_path)
        self.assertEqual(value.shape, (1, 3, 640, 640))
        self.assertEqual(value.dtype, np.float32)
        self.assertAlmostEqual(float(value[0, 0, 320, 320]), 1.0)
        self.assertAlmostEqual(float(value[0, 1, 0, 0]), 114.0 / 255.0)

    def test_yolodecode_probe_documents_nonmatch_without_semantic_drift(self):
        probe = _module("riscy_yolodecode_probe", SAMPLE / "yolodecode_match_probe.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = probe.main(
                [
                    "--source", str(root / "source.onnx"),
                    "--optimized", str(root / "optimized.onnx"),
                    "--report", str(root / "report.json"),
                ]
            )
            report = json.loads((root / "report.json").read_text())
        self.assertEqual(status, 0)
        self.assertEqual(report["yolodecode_fusion_count"], 0)
        self.assertLessEqual(report["max_absolute_difference"], 1e-6)


if __name__ == "__main__":
    unittest.main()
