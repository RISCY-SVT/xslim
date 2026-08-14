# Copyright 2026 RISCY-SVT
"""Deterministic constrained asymmetric INT8 range selection.

This downstream module is model-independent. It simulates ONNX
``QuantizeLinear`` round-to-nearest-even and signed rail saturation while
searching every legal INT8 zero point.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


QUANT_MIN = -128
QUANT_MAX = 127
SILU_REAL_FLOOR = -0.2784645427610738
SUPPORTED_OBJECTIVES = {"minmax", "percentile", "mse", "kl", "constrained-mse"}
SEMANTIC_FLOORS = {"silu": SILU_REAL_FLOOR}


class RangePolicyError(ValueError):
    """Raised when a range policy is invalid or cannot be satisfied."""


@dataclass(frozen=True)
class ConstrainedRangeSpec:
    """Serializable numeric policy for one per-tensor signed activation domain."""

    objective: str = "constrained-mse"
    preserve_zero: bool = True
    required_real_min: Optional[float] = None
    required_real_max: Optional[float] = None
    semantic_floor: Optional[Union[str, float]] = None
    percentile: float = 0.9999
    search_steps: int = 32
    scale_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        objective = str(self.objective).lower().replace("_", "-")
        if objective not in SUPPORTED_OBJECTIVES:
            raise RangePolicyError(
                "objective must be one of " + ", ".join(sorted(SUPPORTED_OBJECTIVES))
            )
        object.__setattr__(self, "objective", objective)
        if not isinstance(self.preserve_zero, bool):
            raise RangePolicyError("preserve_zero must be boolean")
        if not 0.5 < float(self.percentile) <= 1.0:
            raise RangePolicyError("percentile must be in (0.5, 1.0]")
        if not 8 <= int(self.search_steps) <= 512:
            raise RangePolicyError("search_steps must be in [8, 512]")
        if not math.isfinite(float(self.scale_epsilon)) or self.scale_epsilon <= 0:
            raise RangePolicyError("scale_epsilon must be finite and positive")
        for name in ("required_real_min", "required_real_max"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise RangePolicyError(f"{name} must be finite")
        if (
            self.required_real_min is not None
            and self.required_real_max is not None
            and float(self.required_real_min) > float(self.required_real_max)
        ):
            raise RangePolicyError("required_real_min must be <= required_real_max")
        if isinstance(self.semantic_floor, str) and self.semantic_floor.lower() not in SEMANTIC_FLOORS:
            raise RangePolicyError(
                "semantic_floor must be numeric or one of " + ", ".join(sorted(SEMANTIC_FLOORS))
            )
        if isinstance(self.semantic_floor, (float, int)) and not math.isfinite(float(self.semantic_floor)):
            raise RangePolicyError("semantic_floor must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConstrainedRangeSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise RangePolicyError("unknown range-policy fields: " + ", ".join(unknown))
        return cls(**dict(value))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RangeSelection:
    """Selected Q/DQ parameters and their deterministic audit metrics."""

    scale: float
    zero_point: int
    representable_min: float
    representable_max: float
    objective: str
    objective_value: float
    observed_min: float
    observed_max: float
    clipping_fraction: float
    rail_fraction: float
    bias: float
    mae: float
    normalized_mae: float
    cosine: float
    constraint_margins: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def quantize_dequantize(
    values: np.ndarray,
    scale: float,
    zero_point: int,
    quant_min: int = QUANT_MIN,
    quant_max: int = QUANT_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ONNX-compatible per-tensor affine quantization."""

    scale = float(scale)
    zero_point = int(zero_point)
    if not math.isfinite(scale) or scale <= 0:
        raise RangePolicyError("scale must be finite and positive")
    if not quant_min <= zero_point <= quant_max:
        raise RangePolicyError("zero_point is outside the quantized dtype")
    array = np.asarray(values)
    quantized = np.clip(np.rint(array.astype(np.float64) / scale + zero_point), quant_min, quant_max).astype(
        np.int8
    )
    dequantized = (quantized.astype(np.float64) - zero_point) * scale
    return dequantized.astype(array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64), quantized


def _semantic_floor(spec: ConstrainedRangeSpec) -> Tuple[Optional[float], Optional[str]]:
    if spec.semantic_floor is None:
        return None, None
    if isinstance(spec.semantic_floor, str):
        name = spec.semantic_floor.lower()
        return SEMANTIC_FLOORS[name], f"semantic_floor:{name}"
    return float(spec.semantic_floor), "semantic_floor:numeric"


def _constraint_bounds(spec: ConstrainedRangeSpec) -> Tuple[float, float, Dict[str, float]]:
    lower = 0.0 if spec.preserve_zero else math.inf
    upper = 0.0 if spec.preserve_zero else -math.inf
    requested: Dict[str, float] = {}
    if spec.required_real_min is not None:
        value = float(spec.required_real_min)
        lower = min(lower, value)
        requested["required_real_min"] = value
    if spec.required_real_max is not None:
        value = float(spec.required_real_max)
        upper = max(upper, value)
        requested["required_real_max"] = value
    semantic_floor, semantic_name = _semantic_floor(spec)
    if semantic_floor is not None:
        lower = min(lower, semantic_floor)
        requested[str(semantic_name)] = semantic_floor
    if not math.isfinite(lower):
        lower = 0.0
    if not math.isfinite(upper):
        upper = 0.0
    return lower, upper, requested


def _compress_values(values: np.ndarray, max_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        raise RangePolicyError("cannot select a range from an empty tensor")
    if not np.all(np.isfinite(flattened)):
        raise RangePolicyError("observed tensor values must be finite")
    if flattened.size <= max_points:
        return flattened, np.ones(flattened.size, dtype=np.float64)
    lower = float(np.min(flattened))
    upper = float(np.max(flattened))
    if lower == upper:
        return np.asarray([lower], dtype=np.float64), np.asarray([flattened.size], dtype=np.float64)
    counts, edges = np.histogram(flattened, bins=max_points, range=(lower, upper))
    centers = (edges[:-1] + edges[1:]) * 0.5
    mask = counts > 0
    return centers[mask], counts[mask].astype(np.float64)


def _weighted_quantile(points: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(points, kind="mergesort")
    ordered_points = points[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    threshold = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(ordered_points[min(index, ordered_points.size - 1)])


def _required_scale(lower: float, upper: float, zero_point: int, epsilon: float) -> float:
    scale = float(epsilon)
    if lower < 0:
        negative_codes = zero_point - QUANT_MIN
        if negative_codes <= 0:
            return math.inf
        scale = max(scale, -lower / negative_codes)
    if upper > 0:
        positive_codes = QUANT_MAX - zero_point
        if positive_codes <= 0:
            return math.inf
        scale = max(scale, upper / positive_codes)
    return scale


def _kl_loss(codes: np.ndarray, probability: np.ndarray) -> float:
    shifted = codes.astype(np.int16) - QUANT_MIN
    mass = np.bincount(shifted, weights=probability, minlength=256)
    occupied = np.bincount(shifted, weights=(probability > 0).astype(np.float64), minlength=256)
    q = mass[shifted] / np.maximum(occupied[shifted], 1.0)
    mask = probability > 0
    return float(np.sum(probability[mask] * np.log(np.maximum(probability[mask], 1.0e-30) / np.maximum(q[mask], 1.0e-30))))


def _search(
    points: np.ndarray,
    weights: np.ndarray,
    spec: ConstrainedRangeSpec,
    observed_min: float,
    observed_max: float,
) -> RangeSelection:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise RangePolicyError("histogram has no observations")
    probability = weights / weight_sum
    required_lower, required_upper, requested = _constraint_bounds(spec)

    tail = (1.0 - float(spec.percentile)) * 0.5
    percentile_lower = _weighted_quantile(points, weights, tail)
    percentile_upper = _weighted_quantile(points, weights, 1.0 - tail)

    if spec.objective == "minmax":
        search_lower = min(required_lower, observed_min)
        search_upper = max(required_upper, observed_max)
    elif spec.objective == "percentile":
        search_lower = min(required_lower, percentile_lower)
        search_upper = max(required_upper, percentile_upper)
    else:
        search_lower = required_lower
        search_upper = required_upper

    best_key: Optional[Tuple[float, ...]] = None
    best_scale = 0.0
    best_zp = 0
    best_objective = math.inf

    quantile_levels = sorted({spec.percentile, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0})
    quantile_ranges = []
    for level in quantile_levels:
        qtail = (1.0 - level) * 0.5
        quantile_ranges.append(
            (
                _weighted_quantile(points, weights, qtail),
                _weighted_quantile(points, weights, 1.0 - qtail),
            )
        )

    for zero_point in range(QUANT_MIN, QUANT_MAX + 1):
        minimum_scale = _required_scale(search_lower, search_upper, zero_point, spec.scale_epsilon)
        if not math.isfinite(minimum_scale):
            continue
        full_scale = _required_scale(
            min(required_lower, observed_min),
            max(required_upper, observed_max),
            zero_point,
            spec.scale_epsilon,
        )
        if not math.isfinite(full_scale):
            continue
        candidate_scales = {minimum_scale, full_scale}
        for qlower, qupper in quantile_ranges:
            candidate = _required_scale(
                min(required_lower, qlower),
                max(required_upper, qupper),
                zero_point,
                spec.scale_epsilon,
            )
            if math.isfinite(candidate):
                candidate_scales.add(candidate)
        lower_scale = max(spec.scale_epsilon, min(candidate_scales))
        upper_scale = max(lower_scale, max(candidate_scales) * 1.125)
        if upper_scale > lower_scale * (1.0 + 1.0e-12):
            candidate_scales.update(np.geomspace(lower_scale, upper_scale, int(spec.search_steps)).tolist())
        scales = np.asarray(sorted({float(max(item, minimum_scale)) for item in candidate_scales}), dtype=np.float64)

        raw_codes = np.rint(points[None, :] / scales[:, None] + zero_point)
        codes = np.clip(raw_codes, QUANT_MIN, QUANT_MAX)
        reconstructed = (codes - zero_point) * scales[:, None]
        error = reconstructed - points[None, :]
        mse = np.sum(error * error * probability[None, :], axis=1)
        clipped = np.sum(
            ((raw_codes < QUANT_MIN) | (raw_codes > QUANT_MAX)) * probability[None, :],
            axis=1,
        )

        for index, scale in enumerate(scales):
            if spec.objective == "kl":
                objective_value = _kl_loss(codes[index].astype(np.int16), probability)
                primary = objective_value
            elif spec.objective == "percentile":
                allowed = 1.0 - float(spec.percentile)
                excess = max(0.0, float(clipped[index]) - allowed)
                objective_value = float(mse[index])
                primary = excess
            else:
                objective_value = float(mse[index])
                primary = objective_value
            width = float((QUANT_MAX - QUANT_MIN) * scale)
            if spec.objective == "percentile":
                key = (primary, objective_value, width, float(scale), float(zero_point))
            else:
                key = (primary, float(clipped[index]), width, float(scale), float(zero_point))
            if best_key is None or key < best_key:
                best_key = key
                best_scale = float(scale)
                best_zp = int(zero_point)
                best_objective = float(objective_value)

    if best_key is None:
        raise RangePolicyError("no legal signed INT8 scale/zero-point pair satisfies the constraints")

    representable_min = (QUANT_MIN - best_zp) * best_scale
    representable_max = (QUANT_MAX - best_zp) * best_scale
    reconstructed, quantized = quantize_dequantize(points, best_scale, best_zp)
    reconstructed = reconstructed.astype(np.float64)
    error = reconstructed - points
    clipping_fraction = float(
        np.sum(weights[(points < representable_min) | (points > representable_max)]) / weight_sum
    )
    rail_fraction = float(np.sum(weights[(quantized == QUANT_MIN) | (quantized == QUANT_MAX)]) / weight_sum)
    bias = float(np.sum(error * weights) / weight_sum)
    mae = float(np.sum(np.abs(error) * weights) / weight_sum)
    mean_absolute = float(np.sum(np.abs(points) * weights) / weight_sum)
    normalized_mae = mae / max(mean_absolute, spec.scale_epsilon)
    dot = float(np.sum(points * reconstructed * weights))
    lhs = float(np.sum(points * points * weights))
    rhs = float(np.sum(reconstructed * reconstructed * weights))
    cosine = dot / math.sqrt(lhs * rhs) if lhs > 0 and rhs > 0 else (1.0 if lhs == rhs else 0.0)

    margins: Dict[str, float] = {}
    if spec.preserve_zero:
        margins["preserve_zero"] = float(min(-representable_min, representable_max))
    for name, value in requested.items():
        if name == "required_real_max":
            margins[name] = float(representable_max - value)
        else:
            margins[name] = float(value - representable_min)
    if any(value < -max(spec.scale_epsilon, abs(best_scale) * 1.0e-6) for value in margins.values()):
        raise RangePolicyError("selected range violates a required constraint")

    return RangeSelection(
        scale=best_scale,
        zero_point=best_zp,
        representable_min=float(representable_min),
        representable_max=float(representable_max),
        objective=spec.objective,
        objective_value=best_objective,
        observed_min=float(observed_min),
        observed_max=float(observed_max),
        clipping_fraction=clipping_fraction,
        rail_fraction=rail_fraction,
        bias=bias,
        mae=mae,
        normalized_mae=normalized_mae,
        cosine=float(cosine),
        constraint_margins=margins,
    )


def select_qparams(values: Union[np.ndarray, Sequence[float]], spec: ConstrainedRangeSpec) -> RangeSelection:
    """Select deterministic signed per-tensor Q/DQ parameters from samples."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise RangePolicyError("cannot select a range from an empty tensor")
    if not np.all(np.isfinite(array)):
        raise RangePolicyError("observed tensor values must be finite")
    points, weights = _compress_values(array)
    return _search(points, weights, spec, float(np.min(array)), float(np.max(array)))


def select_histogram_qparams(
    histogram: Union[np.ndarray, Sequence[float]],
    observed_min: float,
    observed_max: float,
    spec: ConstrainedRangeSpec,
) -> RangeSelection:
    """Select qparams from an evenly spaced observed-value histogram."""

    counts = np.asarray(histogram, dtype=np.float64).reshape(-1)
    if counts.size == 0 or np.sum(counts) <= 0:
        raise RangePolicyError("histogram has no observations")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise RangePolicyError("histogram counts must be finite and non-negative")
    observed_min = float(observed_min)
    observed_max = float(observed_max)
    if not math.isfinite(observed_min) or not math.isfinite(observed_max):
        raise RangePolicyError("histogram range must be finite")
    if observed_max < observed_min:
        raise RangePolicyError("histogram observed_max must be >= observed_min")
    if observed_max == observed_min:
        points = np.asarray([observed_min], dtype=np.float64)
        weights = np.asarray([float(np.sum(counts))], dtype=np.float64)
    else:
        width = (observed_max - observed_min) / counts.size
        points = observed_min + (np.arange(counts.size, dtype=np.float64) + 0.5) * width
        mask = counts > 0
        points = points[mask]
        weights = counts[mask]
        if points.size > 1024:
            group = int(math.ceil(points.size / 1024))
            padded = int(math.ceil(points.size / group) * group)
            point_pad = np.pad(points, (0, padded - points.size), constant_values=0.0)
            weight_pad = np.pad(weights, (0, padded - weights.size), constant_values=0.0)
            point_matrix = point_pad.reshape(-1, group)
            weight_matrix = weight_pad.reshape(-1, group)
            aggregate_weights = np.sum(weight_matrix, axis=1)
            aggregate_points = np.sum(point_matrix * weight_matrix, axis=1) / np.maximum(aggregate_weights, 1.0)
            nonzero = aggregate_weights > 0
            points = aggregate_points[nonzero]
            weights = aggregate_weights[nonzero]
    return _search(points, weights, spec, observed_min, observed_max)
