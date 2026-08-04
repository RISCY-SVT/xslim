"""Generate a public synthetic graph for the split-producer matcher boundary."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from xslim.onnxslim_pass import optimize_onnx_model


def _tensor(name, dtype, dims, values):
    return helper.make_tensor(name, dtype, dims, values)


def build_model() -> onnx.ModelProto:
    """Build the decode tail with bbox and confidence from distinct producers."""
    bbox = helper.make_tensor_value_info("bbox", TensorProto.FLOAT, [1, 64, 4])
    confidence = helper.make_tensor_value_info(
        "confidence", TensorProto.FLOAT, [1, 8, 4]
    )
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 12, 4])
    initializers = [
        _tensor("reshape0_shape", TensorProto.INT64, [4], [1, 4, 16, 4]),
        _tensor(
            "dfl_weight",
            TensorProto.FLOAT,
            [1, 16, 1, 1],
            [float(index) / 16.0 for index in range(16)],
        ),
        _tensor("reshape1_shape", TensorProto.INT64, [3], [1, 4, 4]),
        _tensor("slice0_starts", TensorProto.INT64, [1], [0]),
        _tensor("slice0_ends", TensorProto.INT64, [1], [2]),
        _tensor("slice1_starts", TensorProto.INT64, [1], [2]),
        _tensor("slice1_ends", TensorProto.INT64, [1], [4]),
        _tensor("slice_axes", TensorProto.INT64, [1], [1]),
        _tensor("slice_steps", TensorProto.INT64, [1], [1]),
        _tensor(
            "sub_const",
            TensorProto.FLOAT,
            [1, 2, 4],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ),
        _tensor(
            "add_const",
            TensorProto.FLOAT,
            [1, 2, 4],
            [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0],
        ),
        _tensor("div_const", TensorProto.FLOAT, [], [2.0]),
        _tensor(
            "mul_const",
            TensorProto.FLOAT,
            [1, 4, 4],
            [8.0, 8.0, 16.0, 16.0] * 4,
        ),
    ]
    nodes = [
        helper.make_node("Identity", ["bbox"], ["bbox_join"], name="bbox_join"),
        helper.make_node(
            "Identity", ["confidence"], ["confidence_join"], name="confidence_join"
        ),
        helper.make_node("Sigmoid", ["confidence_join"], ["scores"], name="sigmoid_0"),
        helper.make_node(
            "Reshape", ["bbox_join", "reshape0_shape"], ["bbox_reshaped"], name="reshape_0"
        ),
        helper.make_node(
            "Transpose", ["bbox_reshaped"], ["bbox_transposed"], name="transpose_0", perm=[0, 2, 1, 3]
        ),
        helper.make_node(
            "Softmax", ["bbox_transposed"], ["bbox_softmax"], name="softmax_0", axis=1
        ),
        helper.make_node(
            "Conv", ["bbox_softmax", "dfl_weight"], ["bbox_conv"], name="conv_0", pads=[0, 0, 0, 0]
        ),
        helper.make_node(
            "Reshape", ["bbox_conv", "reshape1_shape"], ["bbox_xyxy"], name="reshape_1"
        ),
        helper.make_node(
            "Slice",
            ["bbox_xyxy", "slice0_starts", "slice0_ends", "slice_axes", "slice_steps"],
            ["bbox_lo"],
            name="slice_0",
        ),
        helper.make_node(
            "Slice",
            ["bbox_xyxy", "slice1_starts", "slice1_ends", "slice_axes", "slice_steps"],
            ["bbox_hi"],
            name="slice_1",
        ),
        helper.make_node("Sub", ["sub_const", "bbox_lo"], ["corner_lo"], name="sub_0"),
        helper.make_node("Add", ["add_const", "bbox_hi"], ["corner_hi"], name="add_0"),
        helper.make_node("Sub", ["corner_hi", "corner_lo"], ["extent"], name="sub_1"),
        helper.make_node("Add", ["corner_lo", "corner_hi"], ["center_sum"], name="add_1"),
        helper.make_node("Div", ["center_sum", "div_const"], ["center"], name="div_0"),
        helper.make_node("Concat", ["center", "extent"], ["boxes"], name="concat_0", axis=1),
        helper.make_node("Mul", ["boxes", "mul_const"], ["scaled_boxes"], name="mul_0"),
        helper.make_node("Concat", ["scaled_boxes", "scores"], ["output0"], name="concat_1", axis=1),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "separate-producer-decode", [bbox, confidence], [output], initializers),
        opset_imports=[helper.make_opsetid("", 24)],
    )
    return onnx.shape_inference.infer_shapes(model)


def _run(model: onnx.ModelProto, feeds):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    ).run(None, feeds)[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    options = parser.parse_args(argv)

    source = build_model()
    optimized = optimize_onnx_model(source)
    onnx.checker.check_model(source)
    onnx.checker.check_model(optimized)
    feeds = {
        "bbox": np.linspace(-2.0, 2.0, 256, dtype=np.float32).reshape(1, 64, 4),
        "confidence": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 8, 4),
    }
    expected = _run(source, feeds)
    actual = _run(optimized, feeds)
    max_abs = float(np.max(np.abs(expected - actual)))
    fusion_count = sum(
        node.domain == "spacemit_functions" and node.op_type == "YoloDecode"
        for node in optimized.graph.node
    )
    report = {
        "schema_version": 1,
        "source_nodes": len(source.graph.node),
        "optimized_nodes": len(optimized.graph.node),
        "yolodecode_fusion_count": fusion_count,
        "max_absolute_difference": max_abs,
        "matcher_boundary": (
            "bbox and confidence have distinct producers; the split matcher "
            "requires both paths to originate from one Split node"
        ),
        "scope": "matcher applicability only; this is not a score-collapse reproducer",
    }
    options.source.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(source, options.source)
    onnx.save(optimized, options.optimized)
    options.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return int(fusion_count != 0 or max_abs > 1e-6)


if __name__ == "__main__":
    raise SystemExit(main())
