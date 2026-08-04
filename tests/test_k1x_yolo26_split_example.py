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
    def test_config_preserves_six_separate_vendor_boundaries(self):
        config = json.loads((SAMPLE / "config.json").read_text())
        quantization = config["quantization_parameters"]
        boundaries = quantization["truncate_var_names"]
        self.assertEqual(len(boundaries), 6)
        self.assertEqual(len(set(boundaries)), 6)
        self.assertEqual(quantization["precision_level"], 1)
        self.assertEqual(quantization["finetune_level"], 2)
        self.assertTrue(quantization["analysis_enable"])

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
