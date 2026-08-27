"""Opt-in semantic checks for detector outputs with configurable columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from .common import (
    load_preprocess,
    make_feed,
    parse_shape,
    read_paths,
    sha256_array,
    sha256_file,
)


SCHEMA_VERSION = 1


def _column(array: np.ndarray, index: int, label: str) -> np.ndarray:
    if array.ndim == 0:
        raise ValueError(f"{label} column requires an array with rank >= 1")
    width = array.shape[-1]
    resolved = index if index >= 0 else width + index
    if resolved < 0 or resolved >= width:
        raise ValueError(
            f"{label} column {index} is outside output width {width}"
        )
    return array[..., resolved]


def inspect_output(
    value: np.ndarray,
    expected_shape: Tuple[int, ...],
    expected_dtype: str,
    score_column: int,
    class_column: int,
    score_floor: float,
) -> Dict[str, Any]:
    array = np.asarray(value)
    violations: List[str] = []
    if tuple(array.shape) != expected_shape:
        violations.append("unexpected-shape")
    if expected_dtype != "any" and str(array.dtype) != expected_dtype:
        violations.append("unexpected-dtype")
    finite = np.isfinite(array)
    if not finite.all():
        violations.append("non-finite-output")

    scores = _column(array, score_column, "score")
    classes = _column(array, class_column, "class")
    finite_scores = scores[np.isfinite(scores)]
    finite_classes = classes[np.isfinite(classes)]
    if finite_scores.size == 0:
        violations.append("no-finite-scores")
    else:
        if np.count_nonzero(finite_scores) == 0:
            violations.append("all-zero-scores")
        if float(np.min(finite_scores)) == float(np.max(finite_scores)):
            violations.append("constant-scores")
        if np.all(finite_scores <= score_floor):
            violations.append("all-scores-at-or-below-floor")
    unique_classes = np.unique(finite_classes)
    if unique_classes.size == 0:
        violations.append("zero-unique-class-ids")

    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_array(array),
        "non_finite_count": int(array.size - np.count_nonzero(finite)),
        "score_min": (
            float(np.min(finite_scores)) if finite_scores.size else None
        ),
        "score_max": (
            float(np.max(finite_scores)) if finite_scores.size else None
        ),
        "score_mean": (
            float(np.mean(finite_scores)) if finite_scores.size else None
        ),
        "score_stddev": (
            float(np.std(finite_scores)) if finite_scores.size else None
        ),
        "prediction_count": int(np.count_nonzero(finite_scores > score_floor)),
        "unique_class_count": int(unique_classes.size),
        "class_min": (
            float(np.min(unique_classes)) if unique_classes.size else None
        ),
        "class_max": (
            float(np.max(unique_classes)) if unique_classes.size else None
        ),
        "violations": sorted(set(violations)),
        "status": "pass" if not violations else "fail",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xslim-yolo-output-check",
        description=(
            "Run opt-in semantic checks on a configurable detector output. "
            "The command exits nonzero for findings only with --fail-on-violation."
        ),
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--preprocess", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--input-name", default="")
    parser.add_argument("--score-column", required=True, type=int)
    parser.add_argument("--class-column", required=True, type=int)
    parser.add_argument("--expected-shape", required=True)
    parser.add_argument("--expected-dtype", default="float32")
    parser.add_argument("--score-floor", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fail-on-violation", action="store_true")
    return parser


def run(options: argparse.Namespace) -> Dict[str, Any]:
    if options.limit < 0:
        raise ValueError("--limit must be >= 0")
    expected_shape = parse_shape(options.expected_shape)
    preprocess = load_preprocess(options.preprocess)
    paths = read_paths(options.image_list)
    if options.limit:
        paths = paths[: options.limit]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(options.model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [item.name for item in session.get_outputs()]
    if options.output_name not in output_names:
        raise ValueError(
            f"model has no output {options.output_name!r}; available: {output_names}"
        )
    input_names = [item.name for item in session.get_inputs()]

    rows = []
    for index, path in enumerate(paths):
        feed = make_feed(preprocess, path, input_names, options.input_name)
        value = session.run([options.output_name], feed)[0]
        row = inspect_output(
            value,
            expected_shape,
            options.expected_dtype,
            options.score_column,
            options.class_column,
            options.score_floor,
        )
        row.update({"index": index, "input": path.name})
        rows.append(row)

    violations = sorted(
        {finding for row in rows for finding in row["violations"]}
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": "xslim-yolo-output-check",
        "model": options.model.name,
        "model_sha256": sha256_file(options.model),
        "output_name": options.output_name,
        "expected_shape": list(expected_shape),
        "expected_dtype": options.expected_dtype,
        "score_column": options.score_column,
        "class_column": options.class_column,
        "score_floor": options.score_floor,
        "images": len(rows),
        "passes": sum(row["status"] == "pass" for row in rows),
        "failures": sum(row["status"] != "pass" for row in rows),
        "violations": violations,
        "status": "pass" if not violations else "fail",
        "results": rows,
    }
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    options = parser.parse_args(argv)
    try:
        report = run(options)
    except Exception as exc:  # noqa: BLE001
        parser.error(f"{type(exc).__name__}: {exc}")
    return int(options.fail_on_violation and report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
