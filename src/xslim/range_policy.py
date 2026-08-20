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
class RequiredInterval:
    """A real interval that must retain a minimum number of INT8 codes."""

    name: str
    real_min: float
    real_max: float
    minimum_codes: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise RangePolicyError("required interval name must be a non-empty string")
        if not math.isfinite(float(self.real_min)) or not math.isfinite(float(self.real_max)):
            raise RangePolicyError(f"required interval {self.name!r} bounds must be finite")
        if float(self.real_min) > float(self.real_max):
            raise RangePolicyError(f"required interval {self.name!r} real_min must be <= real_max")
        if isinstance(self.minimum_codes, bool) or int(self.minimum_codes) != self.minimum_codes:
            raise RangePolicyError(f"required interval {self.name!r} minimum_codes must be an integer")
        if not 1 <= int(self.minimum_codes) <= QUANT_MAX - QUANT_MIN + 1:
            raise RangePolicyError(f"required interval {self.name!r} minimum_codes must be in [1, 256]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequiredInterval":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise RangePolicyError("unknown required-interval fields: " + ", ".join(unknown))
        return cls(**dict(value))


@dataclass(frozen=True)
class ConstrainedRangeSpec:
    """Serializable numeric policy for one per-tensor signed activation domain."""

    objective: str = "constrained-mse"
    preserve_zero: bool = True
    required_real_min: Optional[float] = None
    required_real_max: Optional[float] = None
    semantic_floor: Optional[Union[str, float]] = None
    required_intervals: Tuple[RequiredInterval, ...] = field(default_factory=tuple)
    minimum_positive_codes: int = 0
    minimum_negative_codes: int = 0
    maximum_clipping_fraction: Optional[float] = None
    maximum_rail_fraction: Optional[float] = None
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
        intervals = tuple(
            item if isinstance(item, RequiredInterval) else RequiredInterval.from_mapping(item)
            for item in self.required_intervals
        )
        names = [item.name for item in intervals]
        if len(names) != len(set(names)):
            raise RangePolicyError("required interval names must be unique")
        object.__setattr__(self, "required_intervals", intervals)
        for name in ("minimum_positive_codes", "minimum_negative_codes"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or not 0 <= int(value) <= 255:
                raise RangePolicyError(f"{name} must be an integer in [0, 255]")
        for name in ("maximum_clipping_fraction", "maximum_rail_fraction"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                raise RangePolicyError(f"{name} must be finite and in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConstrainedRangeSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise RangePolicyError("unknown range-policy fields: " + ", ".join(unknown))
        normalized = dict(value)
        if "required_intervals" in normalized:
            raw_intervals = normalized["required_intervals"]
            if not isinstance(raw_intervals, (list, tuple)):
                raise RangePolicyError("required_intervals must be a list")
            normalized["required_intervals"] = tuple(
                item if isinstance(item, RequiredInterval) else RequiredInterval.from_mapping(item)
                for item in raw_intervals
            )
        return cls(**normalized)

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
    for interval in spec.required_intervals:
        lower = min(lower, float(interval.real_min))
        upper = max(upper, float(interval.real_max))
        requested[f"required_interval:{interval.name}:min"] = float(interval.real_min)
        requested[f"required_interval:{interval.name}:max"] = float(interval.real_max)
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
        points, counts = np.unique(flattened, return_counts=True)
        return points.astype(np.float64), counts.astype(np.float64)
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
    cumulative: np.ndarray = np.cumsum(ordered_weights)
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
    shifted: np.ndarray = codes.astype(np.int16) - QUANT_MIN
    mass = np.bincount(shifted, weights=probability, minlength=256)
    occupied = np.bincount(shifted, weights=(probability > 0).astype(np.float64), minlength=256)
    q = mass[shifted] / np.maximum(occupied[shifted], 1.0)
    mask = probability > 0
    return float(np.sum(probability[mask] * np.log(np.maximum(probability[mask], 1.0e-30) / np.maximum(q[mask], 1.0e-30))))


def _interval_code_count(scale: float, zero_point: int, real_min: float, real_max: float) -> int:
    tolerance = max(abs(scale) * 1.0e-9, 1.0e-12)
    first = max(QUANT_MIN, int(math.ceil(real_min / scale + zero_point - tolerance)))
    last = min(QUANT_MAX, int(math.floor(real_max / scale + zero_point + tolerance)))
    return max(0, last - first + 1)


def validate_qparams_contract(
    scale: float,
    zero_point: int,
    spec: ConstrainedRangeSpec,
    *,
    clipping_fraction: Optional[float] = None,
    rail_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate final exported qparams against the complete constrained contract."""

    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise RangePolicyError("final scale must be finite and positive")
    if isinstance(zero_point, bool) or int(zero_point) != zero_point:
        raise RangePolicyError("final zero point must be an integer")
    zero_point = int(zero_point)
    if not QUANT_MIN <= zero_point <= QUANT_MAX:
        raise RangePolicyError("final zero point is outside signed INT8")

    representable_min = (QUANT_MIN - zero_point) * scale
    representable_max = (QUANT_MAX - zero_point) * scale
    tolerance = max(spec.scale_epsilon, scale * 1.0e-6)
    margins: Dict[str, float] = {}
    interval_codes: Dict[str, int] = {}

    if spec.preserve_zero:
        if not representable_min - tolerance <= 0.0 <= representable_max + tolerance:
            raise RangePolicyError("final qparams do not preserve real zero")
        margins["preserve_zero"] = float(min(-representable_min, representable_max))
    if spec.required_real_min is not None:
        margin = float(spec.required_real_min) - representable_min
        margins["required_real_min"] = margin
        if margin < -tolerance:
            raise RangePolicyError("final qparams violate required_real_min")
    if spec.required_real_max is not None:
        margin = representable_max - float(spec.required_real_max)
        margins["required_real_max"] = margin
        if margin < -tolerance:
            raise RangePolicyError("final qparams violate required_real_max")
    semantic_floor, semantic_name = _semantic_floor(spec)
    if semantic_floor is not None:
        margin = semantic_floor - representable_min
        margins[str(semantic_name)] = float(margin)
        if margin < -tolerance:
            raise RangePolicyError("final qparams violate semantic_floor")
    for interval in spec.required_intervals:
        lower_margin = float(interval.real_min) - representable_min
        upper_margin = representable_max - float(interval.real_max)
        code_count = _interval_code_count(scale, zero_point, interval.real_min, interval.real_max)
        margins[f"required_interval:{interval.name}:min"] = lower_margin
        margins[f"required_interval:{interval.name}:max"] = upper_margin
        interval_codes[interval.name] = code_count
        if lower_margin < -tolerance or upper_margin < -tolerance:
            raise RangePolicyError(f"final qparams do not cover required interval {interval.name!r}")
        if code_count < interval.minimum_codes:
            raise RangePolicyError(
                f"required interval {interval.name!r} has {code_count} codes; "
                f"minimum is {interval.minimum_codes}"
            )

    positive_codes = QUANT_MAX - zero_point
    negative_codes = zero_point - QUANT_MIN
    if positive_codes < spec.minimum_positive_codes:
        raise RangePolicyError(
            f"final qparams have {positive_codes} positive codes; minimum is {spec.minimum_positive_codes}"
        )
    if negative_codes < spec.minimum_negative_codes:
        raise RangePolicyError(
            f"final qparams have {negative_codes} negative codes; minimum is {spec.minimum_negative_codes}"
        )
    if spec.maximum_clipping_fraction is not None:
        if clipping_fraction is None:
            raise RangePolicyError("maximum_clipping_fraction requires final observation metrics")
        if not math.isfinite(float(clipping_fraction)) or clipping_fraction > spec.maximum_clipping_fraction + 1.0e-12:
            raise RangePolicyError("final qparams violate maximum_clipping_fraction")
    if spec.maximum_rail_fraction is not None:
        if rail_fraction is None:
            raise RangePolicyError("maximum_rail_fraction requires final observation metrics")
        if not math.isfinite(float(rail_fraction)) or rail_fraction > spec.maximum_rail_fraction + 1.0e-12:
            raise RangePolicyError("final qparams violate maximum_rail_fraction")

    return {
        "representable_min": float(representable_min),
        "representable_max": float(representable_max),
        "constraint_margins": margins,
        "interval_code_counts": interval_codes,
        "positive_codes": int(positive_codes),
        "negative_codes": int(negative_codes),
        "clipping_fraction": None if clipping_fraction is None else float(clipping_fraction),
        "rail_fraction": None if rail_fraction is None else float(rail_fraction),
    }


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
    required_lower, required_upper, _ = _constraint_bounds(spec)

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
        scales: np.ndarray = np.asarray(
            sorted({float(max(item, minimum_scale)) for item in candidate_scales}), dtype=np.float64
        )

        raw_codes = np.rint(points[None, :] / scales[:, None] + zero_point)
        codes = np.clip(raw_codes, QUANT_MIN, QUANT_MAX)
        reconstructed = (codes - zero_point) * scales[:, None]
        error = reconstructed - points[None, :]
        mse = np.sum(error * error * probability[None, :], axis=1)
        clipped = np.sum(
            ((raw_codes < QUANT_MIN) | (raw_codes > QUANT_MAX)) * probability[None, :],
            axis=1,
        )
        railed = np.sum(
            ((codes == QUANT_MIN) | (codes == QUANT_MAX)) * probability[None, :],
            axis=1,
        )

        for index, scale in enumerate(scales):
            try:
                validate_qparams_contract(
                    float(scale),
                    zero_point,
                    spec,
                    clipping_fraction=float(clipped[index]),
                    rail_fraction=float(railed[index]),
                )
            except RangePolicyError:
                continue
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

    contract = validate_qparams_contract(
        best_scale,
        best_zp,
        spec,
        clipping_fraction=clipping_fraction,
        rail_fraction=rail_fraction,
    )

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
        constraint_margins=contract["constraint_margins"],
    )


def evaluate_qparams(
    values: Union[np.ndarray, Sequence[float]],
    scale: float,
    zero_point: int,
    spec: ConstrainedRangeSpec,
) -> RangeSelection:
    """Evaluate existing qparams against samples and the complete policy contract."""

    array: np.ndarray = np.asarray(values, dtype=np.float64).reshape(-1)
    points, weights = _compress_values(array)
    return _evaluate_points(points, weights, scale, zero_point, spec, float(np.min(array)), float(np.max(array)))


def _evaluate_points(
    points: np.ndarray,
    weights: np.ndarray,
    scale: float,
    zero_point: int,
    spec: ConstrainedRangeSpec,
    observed_min: float,
    observed_max: float,
) -> RangeSelection:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise RangePolicyError("histogram has no observations")
    probability = weights / weight_sum
    reconstructed, quantized = quantize_dequantize(points, scale, zero_point)
    reconstructed = reconstructed.astype(np.float64)
    error = reconstructed - points
    representable_min = (QUANT_MIN - int(zero_point)) * float(scale)
    representable_max = (QUANT_MAX - int(zero_point)) * float(scale)
    clipping_fraction = float(np.sum(weights[(points < representable_min) | (points > representable_max)]) / weight_sum)
    rail_fraction = float(np.sum(weights[(quantized == QUANT_MIN) | (quantized == QUANT_MAX)]) / weight_sum)
    mse = float(np.sum(error * error * probability))
    bias = float(np.sum(error * probability))
    mae = float(np.sum(np.abs(error) * probability))
    mean_absolute = float(np.sum(np.abs(points) * probability))
    normalized_mae = mae / max(mean_absolute, spec.scale_epsilon)
    dot = float(np.sum(points * reconstructed * probability))
    lhs = float(np.sum(points * points * probability))
    rhs = float(np.sum(reconstructed * reconstructed * probability))
    cosine = dot / math.sqrt(lhs * rhs) if lhs > 0 and rhs > 0 else (1.0 if lhs == rhs else 0.0)
    contract = validate_qparams_contract(
        scale,
        zero_point,
        spec,
        clipping_fraction=clipping_fraction,
        rail_fraction=rail_fraction,
    )
    return RangeSelection(
        scale=float(scale),
        zero_point=int(zero_point),
        representable_min=float(representable_min),
        representable_max=float(representable_max),
        objective=spec.objective,
        objective_value=mse,
        observed_min=float(observed_min),
        observed_max=float(observed_max),
        clipping_fraction=clipping_fraction,
        rail_fraction=rail_fraction,
        bias=bias,
        mae=mae,
        normalized_mae=normalized_mae,
        cosine=float(cosine),
        constraint_margins=contract["constraint_margins"],
    )


def select_qparams(values: Union[np.ndarray, Sequence[float]], spec: ConstrainedRangeSpec) -> RangeSelection:
    """Select deterministic signed per-tensor Q/DQ parameters from samples."""

    array: np.ndarray = np.asarray(values, dtype=np.float64).reshape(-1)
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

    points, weights, observed_min, observed_max = _histogram_points(
        histogram, observed_min, observed_max
    )
    return _search(points, weights, spec, observed_min, observed_max)


def evaluate_histogram_qparams(
    histogram: Union[np.ndarray, Sequence[float]],
    observed_min: float,
    observed_max: float,
    scale: float,
    zero_point: int,
    spec: ConstrainedRangeSpec,
) -> RangeSelection:
    """Evaluate reconstructed qparams on the observer's deterministic histogram."""

    points, weights, observed_min, observed_max = _histogram_points(
        histogram, observed_min, observed_max
    )
    return _evaluate_points(points, weights, scale, zero_point, spec, observed_min, observed_max)


def _histogram_points(
    histogram: Union[np.ndarray, Sequence[float]],
    observed_min: float,
    observed_max: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    counts: np.ndarray = np.asarray(histogram, dtype=np.float64).reshape(-1)
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
        points: np.ndarray = np.asarray([observed_min], dtype=np.float64)
        weights: np.ndarray = np.asarray([float(np.sum(counts))], dtype=np.float64)
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
    return points, weights, observed_min, observed_max
