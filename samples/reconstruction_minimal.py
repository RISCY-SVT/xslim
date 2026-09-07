# Copyright 2026 RISCY-SVT
"""Synthetic API exercise; no model, dataset, or detector accuracy claim."""

import json
from typing import Mapping

import numpy as np
import torch

from xslim.reconstruction import (
    AdaptiveWeightRounder,
    ReconstructionConfig,
    ReconstructionResult,
    reconstruct_block,
)


def run_example() -> ReconstructionResult:
    generator = torch.Generator(device="cpu").manual_seed(17)
    weight = torch.tensor([[0.49, 0.40]], dtype=torch.float32)
    scale = torch.tensor([1.0], dtype=torch.float32)
    zero_point = torch.zeros(1, dtype=torch.int8)
    rounder = AdaptiveWeightRounder(weight, scale, zero_point)
    train_inputs = [torch.rand((1, 2), generator=generator) for _ in range(4)]
    validation_inputs = [torch.rand((1, 2), generator=generator) for _ in range(2)]

    def teacher_forward(value: torch.Tensor) -> torch.Tensor:
        return value @ weight.T

    def student_forward(
        value: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        activation_drop_probability: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        assert activation_drop_probability == 0.0
        return value @ weights["weight"].T

    result = reconstruct_block(
        {"weight": rounder},
        train_inputs,
        validation_inputs,
        teacher_forward,
        student_forward,
        block_name="synthetic-linear",
        config=ReconstructionConfig(
            seed=17, max_iterations=8, validation_interval=2, patience=2,
        ),
    )
    codes = result.hardened_weights["weight"]
    assert codes.shape == (1, 2) and codes.dtype == np.int8
    output = validation_inputs[0] @ (torch.from_numpy(codes).float() * scale[:, None]).T
    assert output.shape == (1, 1) and output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert np.isfinite(result.final_validation_loss)
    assert 1 <= result.iterations <= 8
    return result


if __name__ == "__main__":
    print(json.dumps(run_example().manifest(), sort_keys=True))
