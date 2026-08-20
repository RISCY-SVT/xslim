# Copyright 2026 RISCY-SVT
"""Determinism and safety contract for all-S8 block reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
import random
import torch
from types import SimpleNamespace

from xslim.optimizer.training import LearnedStepSizePassDecorator
from xslim.reconstruction import (
    AdaptiveWeightRounder,
    ReconstructionConfig,
    ReconstructionError,
    apply_bias_correction,
    compute_bias_correction,
    reconstruct_block,
    stratified_activation_sample,
)


def test_stratified_sampling_is_repeatable_order_independent_and_covers_channels_tiles():
    values = np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8)
    sample_a, indices_a, manifest_a = stratified_activation_sample(
        values, 96, seed=65001, tensor_name="activation_a"
    )
    stratified_activation_sample(values + 1, 32, seed=65001, tensor_name="activation_b")
    sample_b, indices_b, manifest_b = stratified_activation_sample(
        values, 96, seed=65001, tensor_name="activation_a"
    )

    np.testing.assert_array_equal(sample_a, sample_b)
    np.testing.assert_array_equal(indices_a, indices_b)
    assert manifest_a == manifest_b
    assert manifest_a.channel_coverage == 3
    assert manifest_a.spatial_tile_coverage == 16
    assert not np.array_equal(indices_a, np.arange(indices_a.size, dtype=np.uint64))

    _, _, tight_manifest = stratified_activation_sample(
        values[:1, :2, :4, :4], 16, seed=65001, tensor_name="tight_budget"
    )
    assert tight_manifest.channel_coverage == 2
    assert tight_manifest.spatial_tile_coverage == 16


def test_stratified_sampling_handles_detector_scale_without_materializing_python_indices():
    values = np.arange(80 * 80 * 80, dtype=np.float32).reshape(1, 80, 80, 80)
    sampled, indices, manifest = stratified_activation_sample(
        values, 4096, seed=65001, tensor_name="p3-confidence"
    )

    assert sampled.shape == (4096,)
    assert indices.shape == (4096,)
    assert manifest.channel_coverage == 80
    assert manifest.spatial_tile_coverage == 16
    assert len(np.unique(indices)) == 4096


def test_adaptive_rounding_hardens_to_static_signed_int8_codes():
    rounder = AdaptiveWeightRounder(
        torch.tensor([0.49, 0.51, -0.49, -0.51]),
        torch.tensor(1.0),
        torch.tensor(0.0),
    )
    initial = rounder.hardened_codes()
    np.testing.assert_array_equal(initial, np.asarray([0, 1, 0, -1], dtype=np.int8))

    rounder.alpha.data.fill_(20.0)
    hardened = rounder.hardened_codes()
    assert hardened.dtype == np.int8
    assert np.all(hardened >= -128)
    assert np.all(hardened <= 127)
    assert list(rounder.state_dict()) == ["alpha", "weight", "scale", "zero_point", "floor", "initial_alpha"]


def _run_correlated_reconstruction():
    source_weight = torch.tensor([0.49, 0.40], dtype=torch.float32)
    rounder = AdaptiveWeightRounder(source_weight, torch.tensor(1.0), torch.tensor(0.0))
    inputs = [torch.tensor([1.0, 1.0], dtype=torch.float32) for _ in range(8)]

    def teacher_forward(value):
        return (value * source_weight).sum().reshape(1)

    def student_forward(value, weights, activation_drop_probability, generator):
        assert activation_drop_probability in {0.0, 0.5}
        assert isinstance(generator, torch.Generator)
        return (value * weights["w"]).sum().reshape(1)

    result = reconstruct_block(
        {"w": rounder},
        inputs,
        inputs[:2],
        teacher_forward,
        student_forward,
        block_name="correlated",
        config=ReconstructionConfig(
            seed=65001,
            max_iterations=80,
            validation_interval=1,
            patience=20,
            learning_rate=0.1,
            rounding_regularization=0.001,
        ),
    )
    return result


def test_reconstruction_uses_heldout_best_checkpoint_and_is_deterministic():
    first = _run_correlated_reconstruction()
    second = _run_correlated_reconstruction()

    assert first.rolled_back is False
    assert first.final_validation_loss < first.initial_validation_loss
    np.testing.assert_array_equal(first.hardened_weights["w"], np.asarray([1, 0], dtype=np.int8))
    np.testing.assert_array_equal(first.hardened_weights["w"], second.hardened_weights["w"])
    assert first.manifest() == second.manifest()


def test_reconstruction_rolls_back_when_validation_does_not_improve():
    rounder = AdaptiveWeightRounder(torch.tensor([0.1]), torch.tensor(1.0), torch.tensor(0.0))
    inputs = [torch.tensor([1.0])]

    result = reconstruct_block(
        {"w": rounder},
        inputs,
        inputs,
        lambda value: torch.zeros(1),
        lambda value, weights, probability, generator: value * weights["w"],
        block_name="rollback",
        config=ReconstructionConfig(max_iterations=2, validation_interval=1, patience=1),
    )

    assert result.rolled_back is True
    assert result.stop_reason == "rollback-validation-no-improvement"
    np.testing.assert_array_equal(result.hardened_weights["w"], np.asarray([0], dtype=np.int8))


def test_reconstruction_detaches_teacher_graph_and_rejects_empty_outputs():
    teacher_weight = torch.tensor([0.4], requires_grad=True)
    rounder = AdaptiveWeightRounder(teacher_weight.detach(), torch.tensor(1.0), torch.tensor(0.0))
    inputs = [torch.tensor([1.0])]

    result = reconstruct_block(
        {"w": rounder},
        inputs,
        inputs,
        lambda value: value * teacher_weight,
        lambda value, weights, probability, generator: value * weights["w"],
        block_name="detached-teacher",
        config=ReconstructionConfig(max_iterations=2, validation_interval=1, patience=1),
    )

    assert result.iterations >= 1
    assert teacher_weight.grad is None

    with pytest.raises(ReconstructionError, match="non-empty tensor output set"):
        reconstruct_block(
            {"w": rounder},
            inputs,
            inputs,
            lambda value: [],
            lambda value, weights, probability, generator: [],
            block_name="empty-output",
            config=ReconstructionConfig(max_iterations=1, validation_interval=1, patience=1),
        )


def test_bias_correction_is_channelwise_topology_preserving_and_finite():
    teacher = [np.asarray([[[[3.0]], [[5.0]]]], dtype=np.float32)]
    student = [np.asarray([[[[2.0]], [[7.0]]]], dtype=np.float32)]
    correction = compute_bias_correction(teacher, student)
    np.testing.assert_array_equal(correction, np.asarray([1.0, -2.0]))
    updated = apply_bias_correction(np.asarray([0.5, 1.5], dtype=np.float32), correction)
    np.testing.assert_array_equal(updated, np.asarray([1.5, -0.5], dtype=np.float32))
    with pytest.raises(ReconstructionError, match="floating-point bias initializer"):
        apply_bias_correction(np.asarray([1, 2], dtype=np.int32), correction)


@pytest.mark.parametrize("probability", [-0.1, 0.1, 1.0])
def test_activation_drop_is_bounded_to_predeclared_modes(probability):
    with pytest.raises(ReconstructionError, match="0.0 or 0.5"):
        ReconstructionConfig(activation_drop_probability=probability)


def test_reconstruction_rejects_unsigned_or_asymmetric_weight_contract():
    with pytest.raises(ReconstructionError, match="signed INT8"):
        AdaptiveWeightRounder(torch.ones(1), torch.ones(1), torch.zeros(1), quant_min=0, quant_max=255)
    with pytest.raises(ReconstructionError, match="symmetric"):
        AdaptiveWeightRounder(torch.ones(1), torch.ones(1), torch.ones(1))


def test_explicit_lsq_seed_derives_order_from_block_identity():
    block = SimpleNamespace(rps=[SimpleNamespace(name="block/op")])
    optimizer = LearnedStepSizePassDecorator(is_scale_trainable=True, lr=0.001, seed=65001)
    observed = optimizer._sample_order(block, 64)
    repeated = LearnedStepSizePassDecorator(is_scale_trainable=True, lr=0.001, seed=65001)
    other = SimpleNamespace(rps=[SimpleNamespace(name="other/op")])

    assert observed == repeated._sample_order(block, 64)
    assert observed != optimizer._sample_order(other, 64)
    assert observed == optimizer._sample_order(block, 64)
    assert optimizer.sample_order_manifest["block/op"]["sample_order_sha256"]


def test_explicit_lsq_block_order_is_independent_of_processing_order():
    first = SimpleNamespace(rps=[SimpleNamespace(name="first/op")])
    second = SimpleNamespace(rps=[SimpleNamespace(name="second/op")])
    forward = LearnedStepSizePassDecorator(is_scale_trainable=True, lr=0.001, seed=65001)
    reverse = LearnedStepSizePassDecorator(is_scale_trainable=True, lr=0.001, seed=65001)

    first_forward = forward._sample_order(first, 64)
    second_forward = forward._sample_order(second, 64)
    second_reverse = reverse._sample_order(second, 64)
    first_reverse = reverse._sample_order(first, 64)

    assert first_forward == first_reverse
    assert second_forward == second_reverse


def test_legacy_lsq_clones_advanced_process_rng_without_mutating_it():
    block = SimpleNamespace(rps=[SimpleNamespace(name="block/op")])
    original_state = random.getstate()
    try:
        random.seed(65001)
        random.sample(range(50), 50)
        expected_rng = random.Random()
        expected_rng.setstate(random.getstate())
        expected = list(range(64))
        expected_rng.shuffle(expected)
        state_before_optimizer = random.getstate()

        optimizer = LearnedStepSizePassDecorator(
            is_scale_trainable=True,
            lr=0.001,
        )
        observed = optimizer._sample_order(block, 64)

        assert observed == expected
        assert random.getstate() == state_before_optimizer
        assert optimizer.sample_order_manifest["block/op"]["seed_source"] == "inherited-process-state"
    finally:
        random.setstate(original_state)
