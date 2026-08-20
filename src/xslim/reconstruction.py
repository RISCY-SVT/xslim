# Copyright 2026 RISCY-SVT
"""Deterministic, model-independent all-S8 block reconstruction primitives."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch


class ReconstructionError(ValueError):
    """Raised when a reconstruction contract is invalid or cannot be satisfied."""


@dataclass(frozen=True)
class StratifiedSampleManifest:
    tensor_name: str
    tensor_shape: Tuple[int, ...]
    requested_samples: int
    selected_samples: int
    channel_coverage: int
    channel_count: int
    spatial_tile_coverage: int
    spatial_tile_count: int
    selected_index_sha256: str
    policy: str = "channel-spatial-tile-hash-v2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionConfig:
    seed: int = 65001
    max_iterations: int = 200
    validation_interval: int = 10
    patience: int = 5
    learning_rate: float = 1.0e-2
    rounding_regularization: float = 1.0e-2
    bias_error_weight: float = 0.1
    rank_loss_weight: float = 0.0
    rank_margin: float = 0.0
    activation_drop_probability: float = 0.0
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ReconstructionError("max_iterations must be positive")
        if self.validation_interval <= 0:
            raise ReconstructionError("validation_interval must be positive")
        if self.patience <= 0:
            raise ReconstructionError("patience must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ReconstructionError("learning_rate must be finite and positive")
        for name in ("rounding_regularization", "bias_error_weight", "rank_loss_weight", "rank_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ReconstructionError(f"{name} must be finite and non-negative")
        if self.activation_drop_probability not in {0.0, 0.5}:
            raise ReconstructionError("activation_drop_probability must be 0.0 or 0.5")


@dataclass(frozen=True)
class ReconstructionResult:
    hardened_weights: Dict[str, np.ndarray]
    initial_train_loss: float
    initial_validation_loss: float
    final_train_loss: float
    final_validation_loss: float
    best_validation_loss: float
    iterations: int
    stop_reason: str
    rolled_back: bool
    sample_order_sha256: str
    per_weight_round_up_fraction: Dict[str, float] = field(default_factory=dict)

    def manifest(self) -> Dict[str, Any]:
        return {
            "initial_train_loss": self.initial_train_loss,
            "initial_validation_loss": self.initial_validation_loss,
            "final_train_loss": self.final_train_loss,
            "final_validation_loss": self.final_validation_loss,
            "best_validation_loss": self.best_validation_loss,
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "rolled_back": self.rolled_back,
            "sample_order_sha256": self.sample_order_sha256,
            "per_weight_round_up_fraction": dict(sorted(self.per_weight_round_up_fraction.items())),
        }


def _stable_u64(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Return stable vectorized ranking keys without allocating Python integers."""

    mixed = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return cast(np.ndarray, mixed ^ (mixed >> np.uint64(31)))


def stratified_activation_sample(
    values: np.ndarray,
    max_samples: int,
    *,
    seed: int,
    tensor_name: str,
    spatial_tiles: Tuple[int, int] = (4, 4),
) -> Tuple[np.ndarray, np.ndarray, StratifiedSampleManifest]:
    """Select deterministic channel-by-spatial-tile samples without phase bias."""

    array = np.asarray(values)
    if array.ndim < 2:
        raise ReconstructionError("activation sampling requires a batch and channel dimension")
    if array.size == 0 or max_samples <= 0:
        raise ReconstructionError("activation sampling requires non-empty data and a positive sample budget")
    if not np.all(np.isfinite(array)):
        raise ReconstructionError("activation sampling rejects non-finite values")
    batch, channels = int(array.shape[0]), int(array.shape[1])
    height = int(array.shape[-2]) if array.ndim >= 4 else 1
    width = int(array.shape[-1]) if array.ndim >= 3 else 1
    tile_h, tile_w = map(int, spatial_tiles)
    if tile_h <= 0 or tile_w <= 0:
        raise ReconstructionError("spatial tile counts must be positive")

    reshaped = array.reshape(batch, channels, -1)
    spatial_size = reshaped.shape[-1]
    tile_ids = np.empty(spatial_size, dtype=np.int64)
    for spatial_index in range(spatial_size):
        if array.ndim >= 4:
            row, column = divmod(spatial_index, width)
            tile_row = min(tile_h - 1, row * tile_h // max(height, 1))
            tile_col = min(tile_w - 1, column * tile_w // max(width, 1))
            tile_ids[spatial_index] = tile_row * tile_w + tile_col
        else:
            tile_ids[spatial_index] = min(
                tile_h * tile_w - 1,
                spatial_index * tile_h * tile_w // spatial_size,
            )
    active_tiles = sorted(map(int, np.unique(tile_ids)))
    channel_order = sorted(
        range(channels),
        key=lambda channel: (_stable_u64("xslim-stratified-channel-v2", seed, tensor_name, channel), channel),
    )
    tile_order = sorted(
        active_tiles,
        key=lambda tile: (_stable_u64("xslim-stratified-tile-v2", seed, tensor_name, tile), tile),
    )

    # A diagonal prefix covers both dimensions as quickly as mathematically
    # possible. Remaining pairs are hash-ranked once, avoiding the quadratic
    # greedy ordering and per-element Python objects used by v1.
    diagonal_count = math.lcm(channels, len(tile_order))
    diagonal = [
        (channel_order[index % channels], tile_order[index % len(tile_order)])
        for index in range(diagonal_count)
    ]
    diagonal_set = set(diagonal)
    remaining = [
        (channel, tile)
        for channel in channel_order
        for tile in tile_order
        if (channel, tile) not in diagonal_set
    ]
    remaining.sort(
        key=lambda item: (_stable_u64("xslim-stratified-stratum-v2", seed, tensor_name, *item), item)
    )
    stratum_order = diagonal + remaining

    limit = min(int(max_samples), array.size)
    max_draw = int(math.ceil(limit / len(stratum_order)))
    batch_offsets = np.arange(batch, dtype=np.uint64) * np.uint64(channels * spatial_size)
    ordered_by_stratum: list[Tuple[Tuple[int, int], np.ndarray]] = []
    for channel, tile in stratum_order:
        positions: np.ndarray = np.flatnonzero(tile_ids == tile).astype(np.uint64)
        candidates = (
            batch_offsets[:, None]
            + np.uint64(channel * spatial_size)
            + positions[None, :]
        ).reshape(-1)
        rank_seed = _stable_u64("xslim-stratified-candidate-v2", seed, tensor_name, channel, tile)
        keys = _splitmix64(candidates, rank_seed)
        order = np.lexsort((candidates, keys))
        ordered_by_stratum.append(((channel, tile), candidates[order[:max_draw]]))

    selected: list[int] = []
    selected_strata: set[Tuple[int, int]] = set()
    draw = 0
    while len(selected) < limit:
        added = False
        for stratum, ordered in ordered_by_stratum:
            if draw < len(ordered):
                selected.append(int(ordered[draw]))
                selected_strata.add(stratum)
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        draw += 1
    indices: np.ndarray = np.asarray(selected, dtype="<u8")
    sampled = array.reshape(-1)[indices.astype(np.int64)]
    selected_channels = {channel for channel, _ in selected_strata}
    selected_tiles = {tile for _, tile in selected_strata}
    manifest = StratifiedSampleManifest(
        tensor_name=tensor_name,
        tensor_shape=tuple(map(int, array.shape)),
        requested_samples=int(max_samples),
        selected_samples=int(indices.size),
        channel_coverage=len(selected_channels),
        channel_count=channels,
        spatial_tile_coverage=len(selected_tiles),
        spatial_tile_count=len(active_tiles),
        selected_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
    )
    return sampled, indices, manifest


class AdaptiveWeightRounder(torch.nn.Module):  # type: ignore[misc]
    """Soft floor/ceil rounding that hardens to ordinary static INT8 weights."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        *,
        channel_axis: int = 0,
        quant_min: int = -128,
        quant_max: int = 127,
    ) -> None:
        super().__init__()
        if not weight.is_floating_point():
            raise ReconstructionError("adaptive rounding requires FP32/FP64 source weights")
        if quant_min != -128 or quant_max != 127:
            raise ReconstructionError("adaptive rounding is constrained to signed INT8")
        if scale.ndim not in {0, 1} or torch.any(~torch.isfinite(scale)) or torch.any(scale <= 0):
            raise ReconstructionError("weight scale must be finite, positive, scalar or per-channel")
        if zero_point.shape != scale.shape or torch.any(zero_point != 0):
            raise ReconstructionError("weight zero point must be symmetric and match scale shape")
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.channel_axis = int(channel_axis)
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("scale", scale.detach().clone().to(weight.dtype))
        self.register_buffer("zero_point", zero_point.detach().clone().to(weight.dtype))
        broadcast_shape = [1] * weight.ndim
        if scale.ndim == 1:
            if scale.numel() != weight.shape[self.channel_axis]:
                raise ReconstructionError("per-channel scale length does not match weight channel axis")
            broadcast_shape[self.channel_axis] = scale.numel()
        normalized = weight / self.scale.reshape(broadcast_shape) + self.zero_point.reshape(broadcast_shape)
        floor = torch.floor(normalized).clamp(quant_min, quant_max)
        residual = (normalized - floor).clamp(1.0e-6, 1.0 - 1.0e-6)
        self.register_buffer("floor", floor)
        initial_alpha = torch.log(residual / (1.0 - residual))
        self.alpha = torch.nn.Parameter(initial_alpha.clone())
        self.register_buffer("initial_alpha", initial_alpha.detach().clone())

    def probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha)

    def codes(self, *, hard: bool = False) -> torch.Tensor:
        probability = self.probabilities()
        decision = (probability >= 0.5).to(probability.dtype) if hard else probability
        return (self.floor + decision).clamp(self.quant_min, self.quant_max)

    def dequantized(self, *, hard: bool = False) -> torch.Tensor:
        broadcast_shape = [1] * self.weight.ndim
        if self.scale.ndim == 1:
            broadcast_shape[self.channel_axis] = self.scale.numel()
        return (self.codes(hard=hard) - self.zero_point.reshape(broadcast_shape)) * self.scale.reshape(
            broadcast_shape
        )

    def regularization(self) -> torch.Tensor:
        probability = self.probabilities()
        return (1.0 - torch.abs(2.0 * probability - 1.0)).mean()

    def hardened_codes(self) -> np.ndarray:
        return cast(np.ndarray, self.codes(hard=True).detach().cpu().numpy().astype(np.int8))


def _as_outputs(value: Any) -> Tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Mapping):
        outputs = tuple(value[name] for name in sorted(value))
        if outputs:
            return outputs
    if isinstance(value, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in value):
        outputs = tuple(value)
        if outputs:
            return outputs
    raise ReconstructionError("block forward callback must return a non-empty tensor output set")


def _output_loss(
    student: Any,
    teacher: Any,
    config: ReconstructionConfig,
) -> torch.Tensor:
    student_outputs = _as_outputs(student)
    teacher_outputs = _as_outputs(teacher)
    if len(student_outputs) != len(teacher_outputs):
        raise ReconstructionError("student and teacher output counts differ")
    total = torch.zeros((), dtype=student_outputs[0].dtype, device=student_outputs[0].device)
    for student_output, teacher_output in zip(student_outputs, teacher_outputs):
        if student_output.shape != teacher_output.shape:
            raise ReconstructionError("student and teacher output shapes differ")
        teacher_reference = teacher_output.detach()
        denominator = teacher_reference.pow(2).mean().clamp_min(1.0e-12)
        total = total + (student_output - teacher_reference).pow(2).mean() / denominator
        total = total + config.bias_error_weight * (student_output.mean() - teacher_reference.mean()).pow(2)
        if config.rank_loss_weight > 0 and teacher_reference.shape[-1] >= 2:
            teacher_delta = teacher_reference[..., 1:] - teacher_reference[..., :-1]
            student_delta = student_output[..., 1:] - student_output[..., :-1]
            direction = torch.sign(teacher_delta)
            rank_loss = torch.relu(config.rank_margin - direction * student_delta).mean()
            total = total + config.rank_loss_weight * rank_loss
    return total


def _sample_order(count: int, config: ReconstructionConfig, block_name: str) -> Tuple[list[int], str]:
    if count <= 0:
        raise ReconstructionError("reconstruction dataset is empty")
    derived = _stable_u64("xslim-reconstruction-order-v1", config.seed, block_name)
    order = list(range(count))
    rng = np.random.default_rng(derived)
    rng.shuffle(order)
    payload = np.asarray(order, dtype="<u8").tobytes()
    return order, hashlib.sha256(payload).hexdigest()


def reconstruct_block(
    rounders: Mapping[str, AdaptiveWeightRounder],
    train_inputs: Sequence[Any],
    validation_inputs: Sequence[Any],
    teacher_forward: Callable[[Any], Any],
    student_forward: Callable[[Any, Mapping[str, torch.Tensor], float, torch.Generator], Any],
    *,
    block_name: str,
    config: ReconstructionConfig,
) -> ReconstructionResult:
    """Optimize adaptive rounding with held-out validation and rollback."""

    if not rounders:
        raise ReconstructionError("at least one adaptive weight rounder is required")
    if not train_inputs or not validation_inputs:
        raise ReconstructionError("both training and held-out validation inputs are required")
    ordered_rounders = {name: rounders[name] for name in sorted(rounders)}
    parameters = [parameter for rounder in ordered_rounders.values() for parameter in rounder.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    order, order_sha256 = _sample_order(len(train_inputs), config, block_name)
    generator = torch.Generator(device=parameters[0].device)
    generator.manual_seed(_stable_u64("xslim-reconstruction-generator-v1", config.seed, block_name))

    teacher_train = [teacher_forward(item) for item in train_inputs]
    teacher_validation = [teacher_forward(item) for item in validation_inputs]

    def evaluate(inputs: Sequence[Any], teachers: Sequence[Any], *, hard: bool) -> float:
        with torch.no_grad():
            weights = {name: rounder.dequantized(hard=hard) for name, rounder in ordered_rounders.items()}
            losses = [
                _output_loss(student_forward(item, weights, 0.0, generator), teacher, config)
                for item, teacher in zip(inputs, teachers)
            ]
            return float(torch.stack(losses).mean().item())

    initial_train = evaluate(train_inputs, teacher_train, hard=True)
    initial_validation = evaluate(validation_inputs, teacher_validation, hard=True)
    best_validation = initial_validation
    best_state = {name: rounder.alpha.detach().clone() for name, rounder in ordered_rounders.items()}
    stale = 0
    iterations = 0
    stop_reason = "max-iterations"
    for iteration in range(config.max_iterations):
        index = order[iteration % len(order)]
        weights = {name: rounder.dequantized(hard=False) for name, rounder in ordered_rounders.items()}
        optimizer.zero_grad()
        student = student_forward(
            train_inputs[index],
            weights,
            config.activation_drop_probability,
            generator,
        )
        loss = _output_loss(student, teacher_train[index], config)
        loss = loss + config.rounding_regularization * torch.stack(
            [rounder.regularization() for rounder in ordered_rounders.values()]
        ).mean()
        if not torch.isfinite(loss):
            raise ReconstructionError("reconstruction produced a non-finite loss")
        loss.backward()
        optimizer.step()
        iterations = iteration + 1
        if iterations % config.validation_interval != 0:
            continue
        validation = evaluate(validation_inputs, teacher_validation, hard=True)
        if validation + config.minimum_improvement < best_validation:
            best_validation = validation
            best_state = {name: rounder.alpha.detach().clone() for name, rounder in ordered_rounders.items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                stop_reason = "early-stopping"
                break

    for name, rounder in ordered_rounders.items():
        rounder.alpha.data.copy_(best_state[name])
    final_train = evaluate(train_inputs, teacher_train, hard=True)
    final_validation = evaluate(validation_inputs, teacher_validation, hard=True)
    rolled_back = final_validation >= initial_validation - config.minimum_improvement
    if rolled_back:
        for rounder in ordered_rounders.values():
            rounder.alpha.data.copy_(rounder.initial_alpha)
        final_train = initial_train
        final_validation = initial_validation
        stop_reason = "rollback-validation-no-improvement"

    hardened = {name: rounder.hardened_codes() for name, rounder in ordered_rounders.items()}
    fractions = {
        name: float((rounder.codes(hard=True) > rounder.floor).to(torch.float32).mean().item())
        for name, rounder in ordered_rounders.items()
    }
    return ReconstructionResult(
        hardened_weights=hardened,
        initial_train_loss=initial_train,
        initial_validation_loss=initial_validation,
        final_train_loss=final_train,
        final_validation_loss=final_validation,
        best_validation_loss=best_validation,
        iterations=iterations,
        stop_reason=stop_reason,
        rolled_back=rolled_back,
        sample_order_sha256=order_sha256,
        per_weight_round_up_fraction=fractions,
    )


def compute_bias_correction(
    teacher_outputs: Sequence[np.ndarray],
    student_outputs: Sequence[np.ndarray],
    *,
    channel_axis: int = 1,
) -> np.ndarray:
    """Compute deterministic channel-wise mean-error correction for an existing bias."""

    if not teacher_outputs or len(teacher_outputs) != len(student_outputs):
        raise ReconstructionError("teacher and student output sets must be non-empty and aligned")
    corrections = []
    for teacher, student in zip(teacher_outputs, student_outputs):
        teacher_array = np.asarray(teacher, dtype=np.float64)
        student_array = np.asarray(student, dtype=np.float64)
        if teacher_array.shape != student_array.shape:
            raise ReconstructionError("teacher and student output shapes differ")
        if not np.all(np.isfinite(teacher_array)) or not np.all(np.isfinite(student_array)):
            raise ReconstructionError("bias correction rejects non-finite outputs")
        reduce_axes = tuple(axis for axis in range(teacher_array.ndim) if axis != channel_axis)
        corrections.append(np.mean(teacher_array - student_array, axis=reduce_axes))
    return cast(np.ndarray, np.mean(np.stack(corrections, axis=0), axis=0))


def apply_bias_correction(existing_bias: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Fold a finite correction into an existing bias without changing topology."""

    bias = np.asarray(existing_bias)
    delta = np.asarray(correction)
    if bias.shape != delta.shape:
        raise ReconstructionError("bias and correction shapes differ")
    if not np.issubdtype(bias.dtype, np.floating):
        raise ReconstructionError("bias correction requires an existing floating-point bias initializer")
    if not np.all(np.isfinite(bias)) or not np.all(np.isfinite(delta)):
        raise ReconstructionError("bias correction values must be finite")
    return cast(np.ndarray, (bias.astype(np.float64) + delta.astype(np.float64)).astype(bias.dtype))
