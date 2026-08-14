# Copyright 2026 RISCY-SVT
"""Fail-closed structural audit for the SpacemiT split-model S8-QDQ contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


PROFILE_NAME = "spacemit_k1x_s8_qdq_split_v1"


class ProfileValidationError(ValueError):
    """Raised when a graph violates the declared structural profile."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def _dtype_name(elem_type: int) -> str:
    return str(onnx.helper.tensor_dtype_to_np_dtype(elem_type).name)


def _shape(value: onnx.ValueInfoProto) -> List[Optional[int]]:
    result: List[Optional[int]] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(None)
    return result


def _initializer_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def _producer_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def _consumer_map(model: onnx.ModelProto) -> Dict[str, List[Tuple[onnx.NodeProto, int]]]:
    result: Dict[str, List[Tuple[onnx.NodeProto, int]]] = {}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            result.setdefault(name, []).append((node, index))
    return result


def _check_output_contract(model: onnx.ModelProto, contract: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if len(contract) != 6:
        raise ProfileValidationError("output contract must describe exactly six split outputs")
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        raise ProfileValidationError(f"shape inference failed: {exc}") from exc
    outputs = list(inferred.graph.output)
    if len(outputs) != 6:
        raise ProfileValidationError(f"model has {len(outputs)} outputs; exact six-output contract required")
    observed = [
        {"name": value.name, "shape": _shape(value), "dtype": _dtype_name(value.type.tensor_type.elem_type)}
        for value in outputs
    ]
    expected = [
        {
            "name": str(item["name"]),
            "shape": [None if value is None else int(value) for value in item["shape"]],
            "dtype": str(item.get("dtype", "float32")),
        }
        for item in contract
    ]
    if observed != expected:
        raise ProfileValidationError(
            "output contract mismatch: expected {} observed {}".format(
                json.dumps(expected, sort_keys=True), json.dumps(observed, sort_keys=True)
            )
        )
    return observed


def validate_profile(
    model_path: Path,
    output_contract: Sequence[Mapping[str, Any]],
    *,
    tail_path: Optional[Path] = None,
    expected_tail_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate graph structure without making any provider-placement claim."""

    model_path = Path(model_path)
    model = onnx.load(model_path, load_external_data=True)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ProfileValidationError(f"ONNX checker failed: {exc}") from exc
    initializers = _initializer_map(model)
    producers = _producer_map(model)
    consumers = _consumer_map(model)

    qlinear_nodes = [node for node in model.graph.node if node.op_type.startswith("QLinear")]
    if qlinear_nodes:
        raise ProfileValidationError(
            "QLinear operators are forbidden: " + ", ".join(node.name or node.op_type for node in qlinear_nodes)
        )

    qdq_nodes = [node for node in model.graph.node if node.op_type in {"QuantizeLinear", "DequantizeLinear"}]
    if not qdq_nodes:
        raise ProfileValidationError("signed QDQ graph contains no QuantizeLinear/DequantizeLinear nodes")

    weight_dq: Dict[str, onnx.NodeProto] = {}
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    for conv in conv_nodes:
        if len(conv.input) < 2:
            raise ProfileValidationError(f"Conv {conv.name!r} has no weight input")
        producer = producers.get(conv.input[1])
        if producer is None or producer.op_type != "DequantizeLinear":
            raise ProfileValidationError(f"Conv {conv.name!r} weight is not supplied through signed DQ")
        weight_dq[producer.name or producer.output[0]] = producer

    implicit_uint8 = []
    unsigned = []
    non_initializer_qparams = []
    activation_sites = []
    weight_sites = []
    weight_node_ids = {id(node) for node in weight_dq.values()}
    for node in qdq_nodes:
        if len(node.input) < 2 or node.input[1] not in initializers:
            non_initializer_qparams.append(node.name or node.output[0])
            continue
        scale = np.asarray(initializers[node.input[1]])
        if len(node.input) < 3 or not node.input[2]:
            implicit_uint8.append(node.name or node.output[0])
            continue
        if node.input[2] not in initializers:
            non_initializer_qparams.append(node.name or node.output[0])
            continue
        zero_point = np.asarray(initializers[node.input[2]])
        if zero_point.dtype == np.uint8:
            unsigned.append(node.name or node.output[0])
        elif zero_point.dtype != np.int8:
            raise ProfileValidationError(
                f"Q/DQ zero point for {node.name or node.output[0]!r} has dtype {zero_point.dtype}, expected INT8"
            )
        if id(node) in weight_node_ids:
            axis = _attributes(node).get("axis")
            if axis != 0:
                raise ProfileValidationError(
                    f"Conv weight DQ {node.name!r} must declare per-channel axis=0"
                )
            if scale.ndim != 1 or zero_point.ndim != 1 or scale.size != zero_point.size:
                raise ProfileValidationError(f"Conv weight DQ {node.name!r} is not per-channel")
            if np.any(zero_point != 0):
                raise ProfileValidationError(f"Conv weight DQ {node.name!r} is not symmetric")
            weight_sites.append(node.name or node.output[0])
        else:
            if scale.size != 1 or zero_point.size != 1:
                raise ProfileValidationError(
                    f"activation Q/DQ {node.name or node.output[0]!r} is not per-tensor"
                )
            activation_sites.append(node.name or node.output[0])

    if implicit_uint8 or unsigned:
        raise ProfileValidationError(
            "UINT8 zero points are forbidden: " + ", ".join(sorted(implicit_uint8 + unsigned))
        )
    if non_initializer_qparams:
        raise ProfileValidationError(
            "Q/DQ scale and zero point must be static initializers: " + ", ".join(sorted(non_initializer_qparams))
        )

    kernel_valid = 0
    for conv in conv_nodes:
        attributes = _attributes(conv)
        kernel_shape = attributes.get("kernel_shape")
        if not kernel_shape or any(int(value) <= 0 for value in kernel_shape):
            raise ProfileValidationError(f"Conv {conv.name!r} lacks explicit valid kernel_shape")
        weight_node = producers[conv.input[1]]
        quantized_weight = initializers.get(weight_node.input[0])
        if quantized_weight is None:
            raise ProfileValidationError(f"Conv {conv.name!r} quantized weight is not an initializer")
        if list(map(int, kernel_shape)) != list(map(int, quantized_weight.shape[-2:])):
            raise ProfileValidationError(f"Conv {conv.name!r} kernel_shape does not match weight shape")
        kernel_valid += 1

    fp16_initializers = [item.name for item in model.graph.initializer if item.data_type == TensorProto.FLOAT16]
    if fp16_initializers:
        raise ProfileValidationError("FP16 initializers are forbidden by the all-S8 profile")

    outputs = _check_output_contract(model, output_contract)
    tail_identity = None
    if tail_path is not None:
        tail_path = Path(tail_path)
        if not tail_path.is_file():
            raise ProfileValidationError(f"tail file is missing: {tail_path}")
        tail_identity = _sha256(tail_path)
        if expected_tail_sha256 is not None and tail_identity != expected_tail_sha256:
            raise ProfileValidationError(
                f"tail SHA-256 mismatch: expected {expected_tail_sha256}, observed {tail_identity}"
            )
    elif expected_tail_sha256 is not None:
        raise ProfileValidationError("expected tail SHA-256 was supplied without a tail file")

    return {
        "profile": PROFILE_NAME,
        "passed": True,
        "model_sha256": _sha256(model_path),
        "qlinear_count": 0,
        "quantize_linear_count": sum(node.op_type == "QuantizeLinear" for node in qdq_nodes),
        "dequantize_linear_count": sum(node.op_type == "DequantizeLinear" for node in qdq_nodes),
        "uint8_zero_point_count": 0,
        "activation_per_tensor_sites": len(activation_sites),
        "conv_weight_per_channel_sites": len(weight_sites),
        "conv_kernel_shape": {"total": len(conv_nodes), "valid": kernel_valid},
        "fp16_initializer_count": 0,
        "outputs": outputs,
        "tail_sha256": tail_identity,
        "structural_risk": (
            "A structural pass does not prove SpacemiT provider placement, fusion, kernel selection, latency, or stability."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xslim-spacemit-profile-check")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-contract", required=True, type=Path)
    parser.add_argument("--tail", type=Path)
    parser.add_argument("--tail-sha256")
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract_value = json.loads(args.output_contract.read_text(encoding="utf-8"))
        contract = contract_value.get("outputs", contract_value) if isinstance(contract_value, dict) else contract_value
        if not isinstance(contract, list):
            raise ProfileValidationError("output contract JSON must be a list or an object with an outputs list")
        report = validate_profile(
            args.model,
            contract,
            tail_path=args.tail,
            expected_tail_sha256=args.tail_sha256,
        )
    except Exception as exc:
        report = {"profile": PROFILE_NAME, "passed": False, "error": str(exc)}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
