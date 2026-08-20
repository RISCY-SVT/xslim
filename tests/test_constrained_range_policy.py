# Copyright 2026 RISCY-SVT
"""Numeric contract for the downstream constrained INT8 range selector."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from xslim.range_policy import (
    ConstrainedRangeSpec,
    RangePolicyError,
    _compress_values,
    _kl_loss,
    quantize_dequantize,
    select_qparams,
    validate_qparams_contract,
)


def _run_onnx_qdq(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(values.shape))
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(values.shape))
    initializers = [
        numpy_helper.from_array(np.asarray(scale, dtype=np.float32), name="scale"),
        numpy_helper.from_array(np.asarray(zero_point, dtype=np.int8), name="zero_point"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["quantized"]),
        helper.make_node("DequantizeLinear", ["quantized", "scale", "zero_point"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "qdq", [input_info], [output_info], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": values.astype(np.float32)})[0]


def test_round_half_even_and_onnx_oracle_agree_on_positive_and_negative_ties():
    values = np.asarray([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    expected = _run_onnx_qdq(values, scale=1.0, zero_point=0)
    actual, quantized = quantize_dequantize(values, scale=1.0, zero_point=0)

    np.testing.assert_array_equal(quantized, np.asarray([-2, -2, 0, 0, 2, 2], dtype=np.int8))
    np.testing.assert_array_equal(actual, expected)


def test_rails_saturate_exactly_like_signed_onnx_qdq():
    values = np.asarray([-1000.0, -128.4, 127.4, 1000.0], dtype=np.float32)
    expected = _run_onnx_qdq(values, scale=1.0, zero_point=0)
    actual, quantized = quantize_dequantize(values, scale=1.0, zero_point=0)

    np.testing.assert_array_equal(quantized, np.asarray([-128, -128, 127, 127], dtype=np.int8))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(64, dtype=np.float64),
        np.full(64, 2.25, dtype=np.float64),
        np.full(64, -3.5, dtype=np.float64),
        np.asarray([-0.25, 0.0, 0.25], dtype=np.float64),
    ],
)
def test_degenerate_and_constant_distributions_are_finite_deterministic_and_preserve_zero(values):
    spec = ConstrainedRangeSpec(objective="constrained-mse", preserve_zero=True)
    first = select_qparams(values, spec)
    second = select_qparams(values, spec)

    assert first == second
    assert np.isfinite(first.scale) and first.scale > 0
    assert -128 <= first.zero_point <= 127
    assert first.representable_min <= 0.0 <= first.representable_max
    assert (0.0 / first.scale + first.zero_point) == first.zero_point


def test_outlier_distribution_honors_floor_and_positive_headroom():
    core = np.linspace(-0.25, 3.0, 4096, dtype=np.float64)
    values = np.concatenate([core, np.asarray([90.0, 120.0])])
    spec = ConstrainedRangeSpec(
        objective="constrained-mse",
        preserve_zero=True,
        required_real_min=-0.30,
        required_real_max=8.0,
        percentile=0.999,
    )
    result = select_qparams(values, spec)

    assert result.representable_min <= -0.30
    assert result.representable_max >= 8.0
    assert result.constraint_margins["required_real_min"] >= 0
    assert result.constraint_margins["required_real_max"] >= 0


def test_named_silu_floor_is_opt_in_and_not_applied_globally():
    values = np.linspace(-0.1, 1.0, 1024, dtype=np.float64)
    unconstrained = select_qparams(values, ConstrainedRangeSpec(objective="mse"))
    constrained = select_qparams(
        values,
        ConstrainedRangeSpec(objective="constrained-mse", semantic_floor="silu"),
    )

    assert constrained.representable_min <= -0.2784645427610738
    assert "semantic_floor:silu" in constrained.constraint_margins
    assert unconstrained.constraint_margins.get("semantic_floor:silu") is None


@pytest.mark.parametrize("objective", ["minmax", "percentile", "mse", "kl", "constrained-mse"])
def test_every_supported_objective_is_deterministic_and_respects_constraints(objective):
    rng = np.random.default_rng(65001)
    values = rng.normal(0.2, 1.7, size=2048)
    spec = ConstrainedRangeSpec(
        objective=objective,
        percentile=0.999,
        required_real_min=-2.0,
        required_real_max=4.0,
    )
    first = select_qparams(values, spec)
    second = select_qparams(values, spec)

    assert first == second
    assert first.representable_min <= -2.0
    assert first.representable_max >= 4.0


def test_infeasible_or_invalid_constraints_fail_closed():
    values = np.asarray([0.0, 1.0], dtype=np.float64)
    with pytest.raises(RangePolicyError, match="required_real_min"):
        select_qparams(
            values,
            ConstrainedRangeSpec(required_real_min=2.0, required_real_max=1.0),
        )
    with pytest.raises(RangePolicyError, match="finite"):
        select_qparams(values, ConstrainedRangeSpec(required_real_max=float("inf")))


def test_required_interval_and_signed_code_budgets_are_enforced():
    spec = ConstrainedRangeSpec.from_mapping(
        {
            "required_intervals": [
                {
                    "name": "silu_negative_trough",
                    "real_min": -0.2784645427610738,
                    "real_max": 0.0,
                    "minimum_codes": 3,
                }
            ],
            "minimum_positive_codes": 16,
            "minimum_negative_codes": 3,
        }
    )
    result = select_qparams(np.linspace(-0.25, 4.0, 4096), spec)
    contract = validate_qparams_contract(
        result.scale,
        result.zero_point,
        spec,
        clipping_fraction=result.clipping_fraction,
        rail_fraction=result.rail_fraction,
    )

    assert contract["interval_code_counts"]["silu_negative_trough"] >= 3
    assert contract["positive_codes"] >= 16
    assert contract["negative_codes"] >= 3


def test_final_clipping_and_rail_limits_fail_closed():
    spec = ConstrainedRangeSpec(maximum_clipping_fraction=0.01, maximum_rail_fraction=0.02)
    with pytest.raises(RangePolicyError, match="maximum_clipping_fraction"):
        validate_qparams_contract(0.1, 0, spec, clipping_fraction=0.02, rail_fraction=0.0)
    with pytest.raises(RangePolicyError, match="maximum_rail_fraction"):
        validate_qparams_contract(0.1, 0, spec, clipping_fraction=0.0, rail_fraction=0.03)
    with pytest.raises(RangePolicyError, match="observation metrics"):
        validate_qparams_contract(0.1, 0, spec)


def test_small_array_distribution_preserves_multiplicity_and_order_invariance():
    first_points, first_weights = _compress_values(np.asarray([1.0, 0.0, 0.0, 0.0]))
    second_points, second_weights = _compress_values(np.asarray([0.0, 1.0, 0.0, 0.0]))

    np.testing.assert_array_equal(first_points, np.asarray([0.0, 1.0]))
    np.testing.assert_array_equal(first_weights, np.asarray([3.0, 1.0]))
    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_array_equal(first_weights, second_weights)

    uniform_loss = _kl_loss(np.asarray([0, 0]), np.asarray([0.5, 0.5]))
    multiplicity_loss = _kl_loss(np.asarray([0, 0]), np.asarray([0.75, 0.25]))
    assert uniform_loss == pytest.approx(0.0)
    assert multiplicity_loss > uniform_loss


def test_required_interval_rejects_infeasible_code_resolution():
    spec = ConstrainedRangeSpec.from_mapping(
        {
            "required_real_min": -100.0,
            "required_real_max": 100.0,
            "required_intervals": [
                {"name": "tiny", "real_min": -0.01, "real_max": 0.0, "minimum_codes": 3}
            ],
        }
    )
    with pytest.raises(RangePolicyError, match="no legal signed INT8"):
        select_qparams(np.asarray([-100.0, 0.0, 100.0]), spec)
