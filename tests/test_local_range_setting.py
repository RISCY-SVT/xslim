# Copyright 2026 RISCY-SVT
"""Configuration and selector contract for downstream local range policies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from xslim.optimizer.local_policy import (
    LocalPolicyConflictError,
    build_policy_plan,
)
from xslim.xslim_pipeline import parse_xslim_config
from xslim.xslim_pipeline import quantize_onnx_model


def _minimal_config(custom_setting):
    return {
        "calibration_parameters": {
            "input_parameters": [{"input_name": "input", "input_shape": [1], "dtype": "float32"}]
        },
        "quantization_parameters": {
            "custom_setting": custom_setting,
            "range_policy_manifest_path": "matched.json",
        },
    }


def _policy(tensors, *, objective="constrained-mse", required_max=4.0, name="policy"):
    return {
        "name": name,
        "tensor_names": list(tensors),
        "range_policy": {
            "enabled": True,
            "strict": True,
            "objective": objective,
            "preserve_zero": True,
            "required_real_max": required_max,
            "lock_qparams": True,
        },
    }


def test_nested_range_policy_and_manifest_path_parse_without_losing_types(tmp_path):
    config = _minimal_config([_policy(["tensor_a"])])
    config["quantization_parameters"]["range_policy_manifest_path"] = str(tmp_path / "matched.json")
    setting = parse_xslim_config(config)
    custom = setting.quantization_parameters.custom_setting[0]

    assert custom.name == "policy"
    assert custom.tensor_names == ["tensor_a"]
    assert custom.range_policy.enabled is True
    assert custom.range_policy.objective == "constrained-mse"
    assert custom.range_policy.required_real_max == 4.0
    assert setting.quantization_parameters.range_policy_manifest_path.endswith("matched.json")


def test_exact_selector_matches_only_requested_tensor_and_is_order_invariant():
    available = ["tensor_a", "tensor_b", "tensor_c"]
    first = build_policy_plan(
        [_policy(["tensor_b"], name="b"), _policy(["tensor_a"], name="a")],
        available,
    )
    second = build_policy_plan(
        [_policy(["tensor_a"], name="a"), _policy(["tensor_b"], name="b")],
        available,
    )

    assert first == second
    assert [item.tensor_name for item in first] == ["tensor_a", "tensor_b"]


def test_strict_no_match_fails_closed():
    with pytest.raises(ValueError, match="did not match"):
        build_policy_plan([_policy(["missing"])], ["tensor_a"])


def test_duplicate_names_with_conflicting_policies_fail_closed():
    settings = [
        _policy(["tensor_a"], required_max=4.0, name="first"),
        _policy(["tensor_a"], required_max=8.0, name="second"),
    ]
    with pytest.raises(LocalPolicyConflictError, match="tensor_a"):
        build_policy_plan(settings, ["tensor_a"])


def test_equivalent_overlap_is_deduplicated_deterministically():
    settings = [
        _policy(["tensor_a"], required_max=4.0, name="same"),
        _policy(["tensor_a"], required_max=4.0, name="same"),
    ]
    plan = build_policy_plan(settings, ["tensor_a"])
    assert len(plan) == 1
    assert plan[0].tensor_name == "tensor_a"


def test_existing_custom_setting_fields_remain_available_without_range_policy():
    setting = parse_xslim_config(
        _minimal_config(
            [
                {
                    "input_names": ["input"],
                    "output_names": ["output"],
                    "calibration_type": "percentile",
                    "max_percentile": 0.9995,
                }
            ]
        )
    )
    custom = setting.quantization_parameters.custom_setting[0]
    assert custom.range_policy.enabled is False
    assert custom.calibration_type == "percentile"
    assert custom.max_percentile == 0.9995


def test_policy_plan_manifest_is_json_stable():
    plan = build_policy_plan([_policy(["tensor_b", "tensor_a"], name="stable")], ["tensor_a", "tensor_b"])
    encoded = json.dumps([item.to_dict() for item in plan], sort_keys=True, separators=(",", ":"))
    assert encoded == json.dumps(
        [item.to_dict() for item in reversed(list(reversed(plan)))],
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_tiny_conv_model(path: Path) -> None:
    model_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    model_output = helper.make_tensor_value_info("terminal", TensorProto.FLOAT, [1, 1, 4, 4])
    weight0 = numpy_helper.from_array(np.asarray([[[[0.75]]]], dtype=np.float32), name="weight0")
    weight1 = numpy_helper.from_array(np.asarray([[[[1.25]]]], dtype=np.float32), name="weight1")
    nodes = [
        helper.make_node("Conv", ["input", "weight0"], ["hidden"], name="conv0", kernel_shape=[1, 1]),
        helper.make_node("Relu", ["hidden"], ["activated"], name="relu"),
        helper.make_node("Conv", ["activated", "weight1"], ["terminal"], name="conv1", kernel_shape=[1, 1]),
    ]
    graph = helper.make_graph(nodes, "tiny_local_policy", [model_input], [model_output], [weight0, weight1])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)


def test_selected_qparams_survive_fusion_blockwise_calibration_finetune_and_export(tmp_path):
    model_path = tmp_path / "tiny.onnx"
    calibration_path = tmp_path / "calibration.npy"
    list_path = tmp_path / "calibration.txt"
    manifest_path = tmp_path / "matched.json"
    output_path = tmp_path / "quantized.onnx"
    _write_tiny_conv_model(model_path)
    np.save(calibration_path, np.linspace(-2.0, 3.0, 16, dtype=np.float32).reshape(1, 1, 4, 4))
    list_path.write_text("\n".join([str(calibration_path)] * 10) + "\n", encoding="utf-8")

    config = {
        "model_parameters": {
            "onnx_model": str(model_path),
            "working_dir": str(tmp_path),
            "output_prefix": "quantized",
            "skip_onnxsim": True,
        },
        "calibration_parameters": {
            "calibration_step": 10,
            "calibration_device": "cpu",
            "input_parameters": [
                {
                    "input_name": "input",
                    "input_shape": [1, 1, 4, 4],
                    "file_type": "npy",
                    "data_list_path": str(list_path),
                    "dtype": "float32",
                }
            ],
        },
        "quantization_parameters": {
            "precision_level": 0,
            "finetune_level": 2,
            "analysis_enable": False,
            "range_policy_manifest_path": str(manifest_path),
            "custom_setting": [
                {
                    "name": "terminal-domain",
                    "tensor_names": ["terminal"],
                    "range_policy": {
                        "enabled": True,
                        "strict": True,
                        "objective": "constrained-mse",
                        "required_real_min": -0.5,
                        "required_real_max": 4.0,
                        "lock_qparams": True,
                        "search_steps": 8,
                    },
                }
            ],
        },
    }
    quantize_onnx_model(config, output_path=str(output_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "post-calibration-final"
    assert len(manifest["final_qparams"]) == 1
    final = manifest["final_qparams"][0]
    assert final["tensor_names"] == ["terminal"]
    assert final["locked"] is True
    assert final["representable_min"] <= -0.5
    assert final["representable_max"] >= 4.0

    model = onnx.load(output_path)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    scale = np.float32(final["scale"])
    zero_point = np.int8(final["zero_point"])
    matching_pairs = 0
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or len(node.input) < 3:
            continue
        if node.input[1] not in initializers or node.input[2] not in initializers:
            continue
        if np.asarray(initializers[node.input[1]]).size != 1 or np.asarray(initializers[node.input[2]]).size != 1:
            continue
        if np.asarray(initializers[node.input[1]]).reshape(()).astype(np.float32) == scale and np.asarray(
            initializers[node.input[2]]
        ).reshape(()).astype(np.int8) == zero_point:
            matching_pairs += 1
    assert matching_pairs >= 1
