# Copyright 2026 RISCY-SVT
"""Fail-closed local calibration and constrained-range binding passes."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast

import torch

from ..logger import logger
from ..ppq_decorator import (
    BaseGraph,
    QuantableOperation,
    QuantizationOptimizationPass,
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
    Variable,
    empty_ppq_cache,
    ppq_common,
)
from ..range_policy import (
    ConstrainedRangeSpec,
    RangePolicyError,
    evaluate_histogram_qparams,
    validate_qparams_contract,
)


RANGE_POLICY_DETAIL_KEY = "RISCY_CONSTRAINED_RANGE_POLICY_V1"
RANGE_POLICY_TARGETS_KEY = "RISCY_CONSTRAINED_RANGE_TARGETS_V1"
RANGE_POLICY_LOCK_KEY = "RISCY_CONSTRAINED_RANGE_LOCK_QPARAMS_V1"
RANGE_POLICY_RESULT_KEY = "RISCY_CONSTRAINED_RANGE_RESULT_V1"
RANGE_POLICY_OBSERVATION_KEY = "RISCY_CONSTRAINED_RANGE_OBSERVATION_V1"
LOCAL_POLICY_MANIFEST_KEY = "RISCY_LOCAL_POLICY_MANIFEST_V1"
LOCAL_POLICY_ASSIGNMENT_KEY = "RISCY_LOCAL_POLICY_ASSIGNMENT_V1"
CONSTRAINED_OBSERVER = "constrained_range"
OBSERVER_MAPPING = {
    "default": "xslim",
    "kl": "kl",
    "mse": "mse",
    "minmax": "minmax",
    "percentile": "percentile",
}


class LocalPolicyConflictError(ValueError):
    """Raised when selectors assign incompatible policies to one quant domain."""


@dataclass(frozen=True)
class PolicyPlanEntry:
    tensor_name: str
    policy_name: str
    policy: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _range_mapping(setting: Any) -> Dict[str, Any]:
    if isinstance(setting, Mapping):
        value = setting.get("range_policy") or {}
        return dict(value)
    value = getattr(setting, "range_policy", None)
    if value is None:
        return {}
    return dict(value.__dict__)


def _setting_value(setting: Any, name: str, default: Any = None) -> Any:
    return setting.get(name, default) if isinstance(setting, Mapping) else getattr(setting, name, default)


def _normalized_policy(setting: Any) -> Dict[str, Any]:
    value = _range_mapping(setting)
    spec_fields = {
        key: value[key]
        for key in ConstrainedRangeSpec.__dataclass_fields__
        if key in value
    }
    spec = ConstrainedRangeSpec.from_mapping(spec_fields)
    return {
        "enabled": bool(value.get("enabled", False)),
        "strict": bool(value.get("strict", True)),
        "lock_qparams": bool(value.get("lock_qparams", True)),
        "spec": spec.to_dict(),
    }


def _policy_name(setting: Any, index: int) -> str:
    name = _setting_value(setting, "name")
    if name:
        return str(name)
    if isinstance(setting, Mapping):
        payload = dict(setting)
    else:
        payload = dict(setting.__dict__)
        if hasattr(payload.get("range_policy"), "__dict__"):
            payload["range_policy"] = dict(payload["range_policy"].__dict__)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:12]
    return "local_policy_{}".format(digest)


def build_policy_plan(settings: Sequence[Any], available_tensor_names: Iterable[str]) -> List[PolicyPlanEntry]:
    """Resolve explicit tensor selectors without requiring a PPQ graph."""

    available = set(available_tensor_names)
    assignments: Dict[str, PolicyPlanEntry] = {}
    for index, setting in enumerate(settings or []):
        policy = _normalized_policy(setting)
        if not policy["enabled"]:
            continue
        names = _setting_value(setting, "tensor_names") or []
        name = _policy_name(setting, index)
        if not names:
            raise ValueError(f"range policy {name!r} has no explicit tensor_names")
        missing = sorted(set(names) - available)
        if missing and policy["strict"]:
            raise ValueError(f"range policy {name!r} did not match tensors: {', '.join(missing)}")
        for tensor_name in sorted(set(names) & available):
            entry = PolicyPlanEntry(tensor_name=tensor_name, policy_name=name, policy=policy)
            existing = assignments.get(tensor_name)
            if existing is not None and existing.policy != entry.policy:
                raise LocalPolicyConflictError(
                    f"tensor {tensor_name!r} has conflicting range policies "
                    f"{existing.policy_name!r} and {name!r}"
                )
            if existing is None or name < existing.policy_name:
                assignments[tensor_name] = entry
    return [assignments[name] for name in sorted(assignments)]


def qparams_are_locked(config: TensorQuantizationConfig) -> bool:
    return bool(config.dominated_by.detail.get(RANGE_POLICY_LOCK_KEY, False))


def has_enabled_range_policy(settings: Sequence[Any]) -> bool:
    """Return whether any custom setting explicitly enables constrained finalization."""

    return any(_normalized_policy(setting)["enabled"] for setting in settings or [])


def _bounded_region_variables(graph: BaseGraph, input_names: Sequence[str], output_names: Sequence[str]) -> Set[Variable]:
    if not input_names or not output_names:
        raise ValueError("bounded subgraph selector requires input_names and output_names")
    missing = sorted((set(input_names) | set(output_names)) - set(graph.variables))
    if missing:
        raise ValueError("subgraph selector did not match tensors: {}".format(", ".join(missing)))
    input_vars = {graph.variables[name] for name in input_names if not graph.variables[name].is_parameter}
    output_vars = {graph.variables[name] for name in output_names}
    if not input_vars or not output_vars:
        raise ValueError("bounded subgraph selector has no non-parameter input or output")

    forward_ops: Set[Any] = set()
    frontier = list(input_vars)
    seen_vars = set(frontier)
    while frontier:
        var = frontier.pop()
        if var in output_vars:
            continue
        for op in var.dest_ops:
            if op in forward_ops:
                continue
            forward_ops.add(op)
            for output in op.outputs:
                if output not in seen_vars:
                    seen_vars.add(output)
                    frontier.append(output)

    backward_ops: Set[Any] = set()
    frontier = list(output_vars)
    seen_back = set(frontier)
    while frontier:
        var = frontier.pop()
        if var in input_vars:
            continue
        source = var.source_op
        if source is None or source in backward_ops:
            continue
        backward_ops.add(source)
        for input_var in source.inputs:
            if not input_var.is_parameter and input_var not in seen_back:
                seen_back.add(input_var)
                frontier.append(input_var)

    selected_ops = forward_ops & backward_ops
    if not selected_ops:
        raise ValueError("bounded subgraph selector has no complete input-to-output path")
    reached_outputs = {
        output for op in selected_ops for output in op.outputs if output in output_vars
    }
    if reached_outputs != output_vars:
        missing_outputs = sorted(var.name for var in output_vars - reached_outputs)
        raise ValueError("bounded subgraph selector did not reach outputs: {}".format(", ".join(missing_outputs)))
    variables: Set[Variable] = set(input_vars) | set(output_vars)
    for op in selected_ops:
        variables.update(var for var in op.inputs if not var.is_parameter)
        variables.update(var for var in op.outputs if not var.is_parameter)
    return variables


def _config_records(graph: BaseGraph) -> Tuple[Dict[Variable, List[TensorQuantizationConfig]], Dict[int, List[str]]]:
    by_variable: Dict[Variable, List[TensorQuantizationConfig]] = {}
    root_descriptors: Dict[int, List[str]] = {}
    for operation in graph.topological_sort():
        if not isinstance(operation, QuantableOperation):
            continue
        for index, (variable, config) in enumerate(
            zip(operation.inputs, operation.config.input_quantization_config)
        ):
            if variable.is_parameter:
                continue
            by_variable.setdefault(variable, []).append(config)
            root = config.dominated_by
            root_descriptors.setdefault(id(root), []).append(
                "{}:input:{}:{}".format(operation.name, index, variable.name)
            )
        for index, (variable, config) in enumerate(
            zip(operation.outputs, operation.config.output_quantization_config)
        ):
            if variable.is_parameter:
                continue
            by_variable.setdefault(variable, []).append(config)
            root = config.dominated_by
            root_descriptors.setdefault(id(root), []).append(
                "{}:output:{}:{}".format(operation.name, index, variable.name)
            )
    return by_variable, root_descriptors


def _stable_root_name(root: TensorQuantizationConfig, descriptors: Dict[int, List[str]]) -> str:
    values = descriptors.get(id(root), [])
    if not values:
        raise ValueError("selected quantization root has no stable graph descriptor")
    return sorted(values)[0]


def _write_manifest(path: Optional[str], manifest: Dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_exported_qparams(model: Any, manifest_path: Optional[str]) -> None:
    """Verify selected qparams against the exact Q/DQ initializers in exported ONNX."""

    if not manifest_path:
        return
    destination = Path(manifest_path)
    if not destination.is_file():
        raise RuntimeError(f"range-policy manifest is missing before export audit: {destination}")
    manifest = json.loads(destination.read_text(encoding="utf-8"))
    entries = manifest.get("final_qparams", [])
    if not entries:
        return

    from onnx import numpy_helper

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output if output}
    dq_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
    for entry in entries:
        targets = set(entry["tensor_names"])
        matches = [node for node in dq_nodes if targets.intersection(node.output)]
        if len(matches) != 1:
            raise RuntimeError(
                "exported constrained domain {!r} matched {} DequantizeLinear nodes".format(
                    sorted(targets), len(matches)
                )
            )
        dq_node = matches[0]
        if len(dq_node.input) < 3 or dq_node.input[1] not in initializers or dq_node.input[2] not in initializers:
            raise RuntimeError("exported constrained DQ has dynamic or missing qparams")
        scale_array = initializers[dq_node.input[1]]
        zero_point_array = initializers[dq_node.input[2]]
        if scale_array.size != 1 or zero_point_array.size != 1:
            raise RuntimeError("exported constrained DQ is not per-tensor")
        exported_scale = float(scale_array.reshape(()))
        exported_zero_point = int(zero_point_array.reshape(()))
        expected_scale = float(entry["final_scale"])
        expected_zero_point = int(entry["final_zero_point"])
        if float(torch.tensor(expected_scale, dtype=torch.float32).item()) != exported_scale:
            raise RuntimeError("exported constrained scale differs from finalized manifest")
        if expected_zero_point != exported_zero_point:
            raise RuntimeError("exported constrained zero point differs from finalized manifest")
        q_node = producers.get(dq_node.input[0])
        if q_node is None or q_node.op_type != "QuantizeLinear":
            raise RuntimeError("exported constrained DQ is not paired with QuantizeLinear")
        if len(q_node.input) < 3 or q_node.input[1] not in initializers or q_node.input[2] not in initializers:
            raise RuntimeError("exported constrained Q has dynamic or missing qparams")
        q_scale = initializers[q_node.input[1]]
        q_zero_point = initializers[q_node.input[2]]
        if q_scale.dtype != scale_array.dtype or q_scale.tobytes() != scale_array.tobytes():
            raise RuntimeError("exported constrained Q/DQ scales differ")
        if q_zero_point.dtype != zero_point_array.dtype or q_zero_point.tobytes() != zero_point_array.tobytes():
            raise RuntimeError("exported constrained Q/DQ zero points differ")
        entry["exported_scale"] = exported_scale
        entry["exported_zero_point"] = exported_zero_point
        entry["exported_q_node"] = q_node.name or q_node.output[0]
        entry["exported_dq_node"] = dq_node.name or dq_node.output[0]
        entry["exported_equal"] = True

    manifest["phase"] = "post-export-verified"
    manifest["exported_qparams_equal"] = True
    _write_manifest(manifest_path, manifest)


class LocalPolicyRebindPass(QuantizationOptimizationPass):  # type: ignore[misc]
    """Bind local settings to final post-fusion quantization roots."""

    def __init__(self, custom_setting: Sequence[Any], manifest_path: Optional[str] = None) -> None:
        super().__init__(name="RISCY-SVT Local Policy Rebind Pass")
        self._custom_setting = custom_setting or []
        self._manifest_path = manifest_path

    @empty_ppq_cache  # type: ignore[untyped-decorator]
    def optimize(self, graph: BaseGraph, **kwargs: Any) -> None:
        by_variable, descriptors = _config_records(graph)
        root_assignments: Dict[int, Dict[str, Any]] = {}
        roots: Dict[int, TensorQuantizationConfig] = {}

        sorted_settings = sorted(
            enumerate(self._custom_setting),
            key=lambda item: (
                _policy_name(item[1], item[0]),
                json.dumps(_normalized_policy(item[1]), sort_keys=True),
            ),
        )
        for original_index, setting in sorted_settings:
            policy = _normalized_policy(setting)
            tensor_names = _setting_value(setting, "tensor_names") or []
            if tensor_names:
                missing = sorted(set(tensor_names) - set(graph.variables))
                if missing and policy["enabled"] and policy["strict"]:
                    raise ValueError(
                        "range policy {!r} did not match tensors: {}".format(
                            _policy_name(setting, original_index), ", ".join(missing)
                        )
                    )
                variables = {graph.variables[name] for name in tensor_names if name in graph.variables}
            else:
                input_names = _setting_value(setting, "input_names") or []
                output_names = _setting_value(setting, "output_names") or []
                selector_names = set(input_names) | set(output_names)
                if not policy["enabled"] and not selector_names <= set(graph.variables):
                    variables = set()
                else:
                    variables = _bounded_region_variables(graph, input_names, output_names)

            local_policy = {
                "calibration_type": _setting_value(setting, "calibration_type"),
                "max_percentile": _setting_value(setting, "max_percentile"),
                "range_policy": policy if policy["enabled"] else None,
            }
            if all(value is None for value in local_policy.values()):
                continue
            policy_name = _policy_name(setting, original_index)
            matched = 0
            for variable in sorted(variables, key=lambda item: item.name):
                configs = by_variable.get(variable, [])
                roots_for_var = {id(config.dominated_by): config.dominated_by for config in configs}
                if not roots_for_var:
                    if policy["enabled"] and policy["strict"]:
                        raise ValueError(f"selected tensor {variable.name!r} has no quantization configuration")
                    continue
                if policy["enabled"] and len(roots_for_var) != 1:
                    names = sorted(_stable_root_name(root, descriptors) for root in roots_for_var.values())
                    raise LocalPolicyConflictError(
                        f"selected tensor {variable.name!r} maps to ambiguous quantization roots: " + ", ".join(names)
                    )
                for root_id, root in roots_for_var.items():
                    if root.policy.has_property(QuantizationProperty.PER_CHANNEL):
                        if policy["enabled"]:
                            raise ValueError(f"range policy selected per-channel tensor {variable.name!r}")
                        continue
                    if policy["enabled"] and root.state == QuantizationStates.FP32:
                        raise ValueError(f"range policy selected FP32 tensor {variable.name!r}")
                    assignment = {
                        "policy_name": policy_name,
                        "policy": local_policy,
                        "tensor_names": [variable.name],
                    }
                    existing = root_assignments.get(root_id)
                    if existing is not None and existing["policy"] != assignment["policy"]:
                        raise LocalPolicyConflictError(
                            "quantization root {} has conflicting policies {!r} and {!r}".format(
                                _stable_root_name(root, descriptors), existing["policy_name"], policy_name
                            )
                        )
                    if existing is None:
                        root_assignments[root_id] = assignment
                        roots[root_id] = root
                    else:
                        existing["tensor_names"] = sorted(set(existing["tensor_names"]) | {variable.name})
                        if policy_name < existing["policy_name"]:
                            existing["policy_name"] = policy_name
                    matched += 1
            if policy["enabled"] and matched == 0 and policy["strict"]:
                raise ValueError(f"range policy {policy_name!r} did not match any quantized tensor")

        manifest_entries = []
        for root_id in sorted(root_assignments, key=lambda item: _stable_root_name(roots[item], descriptors)):
            root = roots[root_id]
            assignment = root_assignments[root_id]
            bound_policy = cast(Dict[str, Any], assignment["policy"])
            calibration_type = bound_policy["calibration_type"]
            if calibration_type is not None:
                root.observer_algorithm = OBSERVER_MAPPING[calibration_type]
            max_percentile = bound_policy["max_percentile"]
            if max_percentile is not None:
                root.detail[ppq_common.OBSERVER_PERCENTILE_MANUL_OVERRIDE] = float(max_percentile)
            range_policy = bound_policy["range_policy"]
            if range_policy is not None:
                root.observer_algorithm = CONSTRAINED_OBSERVER
                root.detail[RANGE_POLICY_DETAIL_KEY] = dict(range_policy["spec"])
                root.detail[RANGE_POLICY_TARGETS_KEY] = list(assignment["tensor_names"])
                root.detail[RANGE_POLICY_LOCK_KEY] = bool(range_policy["lock_qparams"])
            root.detail[LOCAL_POLICY_ASSIGNMENT_KEY] = {
                "policy_name": assignment["policy_name"],
                "tensor_names": list(assignment["tensor_names"]),
                "policy": bound_policy,
            }
            entry = {
                "policy_name": assignment["policy_name"],
                "root": _stable_root_name(root, descriptors),
                "tensor_names": list(assignment["tensor_names"]),
                "observer_algorithm": root.observer_algorithm,
                "state": root.state.name,
                "range_policy": range_policy,
            }
            manifest_entries.append(entry)
            logger.info(
                "bound local policy {} to {} ({})".format(
                    assignment["policy_name"],
                    entry["root"],
                    ", ".join(entry["tensor_names"]),
                )
            )

        manifest = {
            "schema": "xslim-local-policy-manifest-v1",
            "phase": "post-fusion-binding",
            "entries": manifest_entries,
        }
        graph._detail[LOCAL_POLICY_MANIFEST_KEY] = manifest
        _write_manifest(self._manifest_path, manifest)


class ConstrainedRangeFinalizePass(QuantizationOptimizationPass):  # type: ignore[misc]
    """Restore and verify selected qparams after block-wise finetuning."""

    def __init__(self, manifest_path: Optional[str] = None) -> None:
        super().__init__(name="RISCY-SVT Constrained Range Finalize Pass")
        self._manifest_path = manifest_path

    @empty_ppq_cache  # type: ignore[untyped-decorator]
    def optimize(self, graph: BaseGraph, **kwargs: Any) -> None:
        _, descriptors = _config_records(graph)
        roots: Dict[int, TensorQuantizationConfig] = {}
        for operation in graph.operations.values():
            if not isinstance(operation, QuantableOperation):
                continue
            for config, _ in operation.config_with_variable:
                root = config.dominated_by
                if RANGE_POLICY_DETAIL_KEY in root.detail:
                    roots[id(root)] = root
        entries = []
        for root in sorted(roots.values(), key=lambda item: _stable_root_name(item, descriptors)):
            assignment = root.detail[LOCAL_POLICY_ASSIGNMENT_KEY]
            result = root.detail.get(RANGE_POLICY_RESULT_KEY)
            is_constrained = RANGE_POLICY_DETAIL_KEY in root.detail
            if is_constrained and not isinstance(result, dict):
                raise RuntimeError(
                    "constrained observer did not emit qparams for {}".format(_stable_root_name(root, descriptors))
                )
            if root.scale is None or root.offset is None:
                raise RuntimeError("constrained observer did not emit final qparams")
            current_scale = (
                root.scale.detach().cpu().reshape(-1)[0].item()
                if isinstance(root.scale, torch.Tensor)
                else root.scale
            )
            current_offset = (
                root.offset.detach().cpu().reshape(-1)[0].item()
                if isinstance(root.offset, torch.Tensor)
                else root.offset
            )
            initial_scale = float(result["scale"])
            initial_zero_point = int(result["zero_point"])
            locked = bool(root.detail.get(RANGE_POLICY_LOCK_KEY, False))
            scale = initial_scale if locked else float(current_scale)
            zero_point = initial_zero_point if locked else int(round(float(current_offset)))
            if not math.isfinite(scale) or scale <= 0:
                raise RuntimeError("constrained observer emitted invalid scale")
            if root.quant_min != -128 or root.quant_max != 127:
                raise RuntimeError("constrained range policy requires signed INT8 [-128, 127]")
            if not root.policy.has_property(QuantizationProperty.PER_TENSOR):
                raise RuntimeError("constrained range policy requires per-tensor quantization")
            if not root.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                raise RuntimeError("constrained range policy requires asymmetric quantization")
            device = root.scale.device if isinstance(root.scale, torch.Tensor) else torch.device("cpu")
            root.scale = torch.tensor(scale, dtype=torch.float32, device=device)
            root.offset = torch.tensor(float(zero_point), dtype=torch.float32, device=device)
            root.state = QuantizationStates.ACTIVATED
            representable_min = (root.quant_min - zero_point) * scale
            representable_max = (root.quant_max - zero_point) * scale
            spec = ConstrainedRangeSpec.from_mapping(root.detail[RANGE_POLICY_DETAIL_KEY])
            observation = root.detail.get(RANGE_POLICY_OBSERVATION_KEY)
            try:
                if isinstance(observation, dict):
                    final_metrics = evaluate_histogram_qparams(
                        observation["histogram"],
                        observation["observed_min"],
                        observation["observed_max"],
                        scale,
                        zero_point,
                        spec,
                    ).to_dict()
                else:
                    final_metrics = validate_qparams_contract(scale, zero_point, spec)
            except (KeyError, RangePolicyError) as exc:
                raise RuntimeError(
                    "final reconstructed qparams violate constrained policy for {}: {}".format(
                        _stable_root_name(root, descriptors), exc
                    )
                ) from exc
            entry = {
                "root": _stable_root_name(root, descriptors),
                "policy_name": assignment["policy_name"],
                "tensor_names": assignment["tensor_names"],
                "observer_algorithm": root.observer_algorithm,
                "initial_scale": initial_scale,
                "initial_zero_point": initial_zero_point,
                "final_scale": scale,
                "final_zero_point": zero_point,
                "scale_delta": scale - initial_scale,
                "zero_point_delta": zero_point - initial_zero_point,
                "selection_source": "observer-locked" if locked else "block-reconstruction",
                "scale": scale,
                "zero_point": zero_point,
                "representable_min": representable_min,
                "representable_max": representable_max,
                "locked": locked,
                "initial_result": result,
                "final_contract": final_metrics,
            }
            entries.append(entry)

        manifest = dict(graph._detail.get(LOCAL_POLICY_MANIFEST_KEY, {}))
        manifest.update(
            {
                "schema": "xslim-local-policy-manifest-v1",
                "phase": "post-calibration-final",
                "final_qparams": entries,
            }
        )
        graph._detail[LOCAL_POLICY_MANIFEST_KEY] = manifest
        _write_manifest(self._manifest_path, manifest)
