"""Audit Q/DQ ranges and observed saturation at selected tensor boundaries."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .common import (
    load_preprocess,
    make_feed,
    read_paths,
    sha256_array,
    sha256_file,
)


SCHEMA_VERSION = 1


@dataclass
class QdqSite:
    requested_name: str
    quantized_name: str
    scale: np.ndarray
    zero_point: np.ndarray
    axis: Optional[int]
    source_op: str


@dataclass
class Accumulator:
    count: int = 0
    below: int = 0
    above: int = 0
    rail_min: int = 0
    rail_max: int = 0
    difference_sum: float = 0.0
    absolute_difference_sum: float = 0.0
    reference_absolute_sum: float = 0.0
    dot_sum: float = 0.0
    reference_square_sum: float = 0.0
    dequant_square_sum: float = 0.0
    reference_min: float = math.inf
    reference_max: float = -math.inf
    histogram: Counter = field(default_factory=Counter)
    image_hashes: List[Dict[str, str]] = field(default_factory=list)


def _attributes(node: onnx.NodeProto) -> Dict[str, Any]:
    return {
        item.name: helper.get_attribute_value(item) for item in node.attribute
    }


def _constant_values(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    values = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        value = _attributes(node).get("value")
        if isinstance(value, onnx.TensorProto):
            values[node.output[0]] = np.asarray(numpy_helper.to_array(value))
    return values


def _find_site(model: onnx.ModelProto, name: str) -> QdqSite:
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for item in node.input:
            consumers.setdefault(item, []).append(node)

    quantize = None
    dequantize = None
    producer = producers.get(name)
    if producer is not None and producer.op_type == "QuantizeLinear":
        quantize = producer
    elif producer is not None and producer.op_type == "DequantizeLinear":
        dequantize = producer
    else:
        q_consumers = [
            node
            for node in consumers.get(name, [])
            if node.op_type == "QuantizeLinear"
        ]
        dq_consumers = [
            node
            for node in consumers.get(name, [])
            if node.op_type == "DequantizeLinear"
        ]
        if len(q_consumers) == 1:
            quantize = q_consumers[0]
        elif len(dq_consumers) == 1:
            dequantize = dq_consumers[0]

    node = quantize or dequantize
    if node is None:
        raise ValueError(f"tensor {name!r} is not adjacent to one Q/DQ site")
    if len(node.input) < 2:
        raise ValueError(f"{node.op_type} for {name!r} has no scale input")
    constants = _constant_values(model)
    if node.input[1] not in constants:
        raise ValueError(f"scale for {name!r} is not a constant initializer")
    scale = np.asarray(constants[node.input[1]])
    if len(node.input) >= 3 and node.input[2]:
        if node.input[2] not in constants:
            raise ValueError(
                f"zero point for {name!r} is not a constant initializer"
            )
        zero_point = np.asarray(constants[node.input[2]])
    else:
        zero_point = np.asarray(0, dtype=np.uint8)
    axis_value = _attributes(node).get("axis")
    axis = int(axis_value) if axis_value is not None and scale.size > 1 else None
    quantized_name = (
        quantize.output[0] if quantize is not None else dequantize.input[0]
    )
    return QdqSite(
        requested_name=name,
        quantized_name=quantized_name,
        scale=scale,
        zero_point=zero_point,
        axis=axis,
        source_op=node.op_type,
    )


def _value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model)
    values = {}
    for item in list(inferred.graph.input) + list(inferred.graph.output) + list(
        inferred.graph.value_info
    ):
        values[item.name] = item
    for item in inferred.graph.initializer:
        values.setdefault(
            item.name,
            helper.make_tensor_value_info(item.name, item.data_type, item.dims),
        )
    return values


def _model_with_outputs(
    model: onnx.ModelProto, output_names: Iterable[str]
) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    existing = {item.name for item in result.graph.output}
    values = _value_info_map(result)
    for name in output_names:
        if name in existing:
            continue
        value = values.get(name)
        if value is None:
            raise ValueError(f"cannot infer type and shape for tensor {name!r}")
        result.graph.output.append(copy.deepcopy(value))
        existing.add(name)
    return result


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _broadcast(parameter: np.ndarray, shape: Tuple[int, ...], axis: Optional[int]):
    value = np.asarray(parameter)
    if value.size == 1:
        return value.reshape(()).astype(np.float64)
    if axis is None:
        raise ValueError("multi-value Q/DQ parameter has no axis")
    resolved_axis = axis if axis >= 0 else len(shape) + axis
    if resolved_axis < 0 or resolved_axis >= len(shape):
        raise ValueError(f"Q/DQ axis {axis} is invalid for shape {shape}")
    if value.size != shape[resolved_axis]:
        raise ValueError(
            f"Q/DQ parameter size {value.size} does not match axis {axis} "
            f"of shape {shape}"
        )
    broadcast_shape = [1] * len(shape)
    broadcast_shape[resolved_axis] = value.size
    return value.reshape(broadcast_shape).astype(np.float64)


def update_accumulator(
    accumulator: Accumulator,
    reference: np.ndarray,
    quantized: np.ndarray,
    site: QdqSite,
    image_name: str,
) -> None:
    if reference.shape != quantized.shape:
        raise ValueError(
            f"shape mismatch for {site.requested_name}: "
            f"float={reference.shape}, quantized={quantized.shape}"
        )
    if not np.issubdtype(quantized.dtype, np.integer):
        raise TypeError(
            f"quantized tensor {site.quantized_name!r} has dtype {quantized.dtype}"
        )
    limits = np.iinfo(quantized.dtype)
    scale = _broadcast(site.scale, quantized.shape, site.axis)
    zero_point = _broadcast(site.zero_point, quantized.shape, site.axis)
    lower = (limits.min - zero_point) * scale
    upper = (limits.max - zero_point) * scale
    ref = reference.astype(np.float64, copy=False)
    q = quantized.astype(np.float64, copy=False)
    dequantized = (q - zero_point) * scale
    difference = dequantized - ref

    accumulator.count += int(ref.size)
    accumulator.below += int(np.count_nonzero(ref < lower))
    accumulator.above += int(np.count_nonzero(ref > upper))
    accumulator.rail_min += int(np.count_nonzero(quantized == limits.min))
    accumulator.rail_max += int(np.count_nonzero(quantized == limits.max))
    accumulator.difference_sum += float(np.sum(difference))
    accumulator.absolute_difference_sum += float(np.sum(np.abs(difference)))
    accumulator.reference_absolute_sum += float(np.sum(np.abs(ref)))
    accumulator.dot_sum += float(np.sum(ref * dequantized))
    accumulator.reference_square_sum += float(np.sum(ref * ref))
    accumulator.dequant_square_sum += float(np.sum(dequantized * dequantized))
    if ref.size:
        accumulator.reference_min = min(
            accumulator.reference_min, float(np.min(ref))
        )
        accumulator.reference_max = max(
            accumulator.reference_max, float(np.max(ref))
        )
    unique, counts = np.unique(quantized, return_counts=True)
    accumulator.histogram.update(
        {int(key): int(count) for key, count in zip(unique, counts)}
    )
    accumulator.image_hashes.append(
        {
            "image": image_name,
            "float": sha256_array(reference),
            "quantized": sha256_array(quantized),
            "dequantized": sha256_array(dequantized),
        }
    )


def _parameter_summary(value: np.ndarray) -> str:
    flat = np.asarray(value).reshape(-1)
    if flat.size <= 16:
        return json.dumps(flat.tolist(), separators=(",", ":"))
    return json.dumps(
        {
            "count": int(flat.size),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "unique": int(np.unique(flat).size),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _result_row(
    float_name: str, site: QdqSite, accumulator: Accumulator
) -> Dict[str, Any]:
    count = accumulator.count
    dtype = np.asarray(site.zero_point).dtype
    limits = np.iinfo(dtype)
    scale = np.asarray(site.scale, dtype=np.float64)
    zero_point = np.asarray(site.zero_point, dtype=np.float64)
    lower = (limits.min - zero_point) * scale
    upper = (limits.max - zero_point) * scale
    denominator = math.sqrt(
        accumulator.reference_square_sum * accumulator.dequant_square_sum
    )
    mae = accumulator.absolute_difference_sum / count if count else math.nan
    reference_mean_abs = accumulator.reference_absolute_sum / count if count else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "float_tensor": float_name,
        "quant_tensor": site.requested_name,
        "raw_quantized_tensor": site.quantized_name,
        "source_op": site.source_op,
        "dtype": str(dtype),
        "axis": "" if site.axis is None else site.axis,
        "granularity": "per-tensor" if site.scale.size == 1 else "per-channel",
        "scale": _parameter_summary(site.scale),
        "zero_point": _parameter_summary(site.zero_point),
        "representable_min": float(np.min(lower)),
        "representable_max": float(np.max(upper)),
        "fp32_min": accumulator.reference_min,
        "fp32_max": accumulator.reference_max,
        "sample_count": count,
        "below_range_count": accumulator.below,
        "below_range_fraction": accumulator.below / count if count else math.nan,
        "above_range_count": accumulator.above,
        "above_range_fraction": accumulator.above / count if count else math.nan,
        "quantized_min_rail_hits": accumulator.rail_min,
        "quantized_max_rail_hits": accumulator.rail_max,
        "dequantized_min_rail_hits": accumulator.rail_min,
        "dequantized_max_rail_hits": accumulator.rail_max,
        "rail_hit_fraction": (
            (accumulator.rail_min + accumulator.rail_max) / count
            if count
            else math.nan
        ),
        "mean_bias": accumulator.difference_sum / count if count else math.nan,
        "mae": mae,
        "normalized_mae": (
            mae / reference_mean_abs if reference_mean_abs else math.nan
        ),
        "cosine": accumulator.dot_sum / denominator if denominator else math.nan,
        "histogram": json.dumps(
            dict(sorted(accumulator.histogram.items())),
            separators=(",", ":"),
        ),
        "per_image_hashes": json.dumps(
            accumulator.image_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _tensor_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) == 1:
            fields = line.split()
        if len(fields) == 1:
            pairs.append((fields[0], fields[0]))
        elif len(fields) == 2:
            pairs.append((fields[0], fields[1]))
        else:
            raise ValueError(
                f"{path}:{line_number}: expected float_name[tab]quant_name"
            )
    if not pairs:
        raise ValueError(f"tensor list is empty: {path}")
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xslim-qdq-boundary-audit")
    parser.add_argument("--float-model", required=True, type=Path)
    parser.add_argument("--quant-model", required=True, type=Path)
    parser.add_argument("--tensor-list", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--preprocess", required=True)
    parser.add_argument("--input-name", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def run(options: argparse.Namespace) -> List[Dict[str, Any]]:
    if options.limit < 0:
        raise ValueError("--limit must be >= 0")
    pairs = _tensor_pairs(options.tensor_list)
    float_model = onnx.load(options.float_model)
    quant_model = onnx.load(options.quant_model)
    sites = {quant: _find_site(quant_model, quant) for _, quant in pairs}
    float_session = _session(
        _model_with_outputs(float_model, [item[0] for item in pairs])
    )
    quant_session = _session(
        _model_with_outputs(
            quant_model, [sites[item[1]].quantized_name for item in pairs]
        )
    )
    float_inputs = [item.name for item in float_session.get_inputs()]
    quant_inputs = [item.name for item in quant_session.get_inputs()]
    if float_inputs != quant_inputs:
        raise ValueError(
            f"model input mismatch: float={float_inputs}, quant={quant_inputs}"
        )
    preprocess = load_preprocess(options.preprocess)
    paths = read_paths(options.image_list)
    if options.limit:
        paths = paths[: options.limit]
    accumulators = {pair: Accumulator() for pair in pairs}
    float_outputs = [item[0] for item in pairs]
    quant_outputs = [sites[item[1]].quantized_name for item in pairs]
    for path in paths:
        feed = make_feed(preprocess, path, float_inputs, options.input_name)
        reference_values = float_session.run(float_outputs, feed)
        quantized_values = quant_session.run(quant_outputs, feed)
        for pair, reference, quantized in zip(
            pairs, reference_values, quantized_values
        ):
            update_accumulator(
                accumulators[pair], reference, quantized, sites[pair[1]], path.name
            )
    rows = [
        _result_row(float_name, sites[quant_name], accumulators[(float_name, quant_name)])
        for float_name, quant_name in pairs
    ]
    options.report.parent.mkdir(parents=True, exist_ok=True)
    with options.report.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    options = parser.parse_args(argv)
    try:
        run(options)
    except Exception as exc:  # noqa: BLE001
        parser.error(f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
