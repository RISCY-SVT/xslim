# Copyright 2026 RISCY-SVT
"""Structural validation tests for the downstream SpacemiT S8-QDQ profile."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from xslim.tools.spacemit_profile import ProfileValidationError, validate_profile


def _build_model(path: Path, *, unsigned=False, kernel_shape=True, qlinear=False) -> list[dict]:
    zp_dtype = np.uint8 if unsigned else np.int8
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    initializers = [
        numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), name="act_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=zp_dtype), name="act_zp"),
        numpy_helper.from_array(np.asarray([2], dtype=np.int8).reshape(1, 1, 1, 1), name="weight_q"),
        numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="weight_scale"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="weight_zp"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "act_scale", "act_zp"], ["input_q"], name="input_q"),
        helper.make_node("DequantizeLinear", ["input_q", "act_scale", "act_zp"], ["input_dq"], name="input_dq"),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zp"],
            ["weight_dq"],
            name="weight_dq",
            axis=0,
        ),
    ]
    conv_kwargs = {"kernel_shape": [1, 1]} if kernel_shape else {}
    nodes.append(helper.make_node("Conv", ["input_dq", "weight_dq"], ["conv"], name="conv", **conv_kwargs))
    if qlinear:
        initializers.extend(
            [
                numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), name="y_scale"),
                numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="y_zp"),
            ]
        )
        nodes.append(
            helper.make_node(
                "QLinearConv",
                ["input_q", "act_scale", "act_zp", "weight_q", "weight_scale", "weight_zp", "y_scale", "y_zp"],
                ["unused_qlinear"],
                name="forbidden_qlinear",
                kernel_shape=[1, 1],
            )
        )

    outputs = []
    contract = []
    for index in range(6):
        scale = f"out_{index}_scale"
        zp = f"out_{index}_zp"
        quantized = f"out_{index}_q"
        name = f"boundary_{index}"
        initializers.extend(
            [
                numpy_helper.from_array(np.asarray(0.2 + index * 0.01, dtype=np.float32), name=scale),
                numpy_helper.from_array(np.asarray(0, dtype=zp_dtype), name=zp),
            ]
        )
        nodes.extend(
            [
                helper.make_node("QuantizeLinear", ["conv", scale, zp], [quantized], name=f"q_{index}"),
                helper.make_node("DequantizeLinear", [quantized, scale, zp], [name], name=f"dq_{index}"),
            ]
        )
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 4, 4]))
        contract.append({"name": name, "shape": [1, 1, 4, 4], "dtype": "float32"})

    graph = helper.make_graph(nodes, "profile", [input_info], outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)
    return contract


def test_valid_signed_split_profile_passes_without_runtime_claims(tmp_path):
    model_path = tmp_path / "valid.onnx"
    tail_path = tmp_path / "tail.onnx"
    tail_path.write_bytes(b"fixed-tail")
    contract = _build_model(model_path)

    report = validate_profile(model_path, contract, tail_path=tail_path)
    assert report["profile"] == "spacemit_k1x_s8_qdq_split_v1"
    assert report["passed"] is True
    assert report["qlinear_count"] == 0
    assert report["uint8_zero_point_count"] == 0
    assert report["conv_kernel_shape"]["valid"] == 1
    assert "provider_placement" not in report


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"unsigned": True}, "UINT8"),
        ({"kernel_shape": False}, "kernel_shape"),
        ({"qlinear": True}, "QLinear"),
    ],
)
def test_profile_fails_closed_on_vendor_contract_violation(tmp_path, kwargs, needle):
    model_path = tmp_path / "invalid.onnx"
    contract = _build_model(model_path, **kwargs)
    with pytest.raises(ProfileValidationError, match=needle):
        validate_profile(model_path, contract)


def test_output_order_shape_and_tail_identity_are_enforced(tmp_path):
    model_path = tmp_path / "valid.onnx"
    tail_path = tmp_path / "tail.onnx"
    tail_path.write_bytes(b"fixed-tail")
    contract = _build_model(model_path)

    wrong_order = list(reversed(contract))
    with pytest.raises(ProfileValidationError, match="output contract"):
        validate_profile(model_path, wrong_order, tail_path=tail_path)

    with pytest.raises(ProfileValidationError, match="tail SHA-256"):
        validate_profile(model_path, contract, tail_path=tail_path, expected_tail_sha256="0" * 64)
