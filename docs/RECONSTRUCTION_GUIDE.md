# Reconstruction Guide

XSlim exposes deterministic, caller-driven layer-local reconstruction
primitives. It does not automatically discover or train a full model.

## Supported Foundation

The generic engine provides:

- deterministic channel/spatial-tile activation sampling;
- explicit global and per-block seeds;
- soft floor/ceil weight rounding with deterministic hardening;
- normalized block-output MSE and mean/bias loss;
- optional pairwise confidence-rank loss supplied by the caller;
- held-out validation, early stopping, best-checkpoint restore, and rollback;
- deterministic bias correction that can be folded into an existing bias;
- ordinary static INT8 output codes with no training operator in ONNX.

The public entry points are `ReconstructionConfig`,
`AdaptiveWeightRounder`, `reconstruct_block`, `stratified_activation_sample`,
`compute_bias_correction`, and `apply_bias_correction` in
`xslim.reconstruction`.

## Executable Synthetic Example

This example exercises the public API on six small CPU tensors with a local
seed and at most eight iterations. It reads no dataset and exports no model.
Run it from the source checkout or extracted source distribution after
[installation](../INSTALL.md):

```bash
python samples/reconstruction_minimal.py
```

Expected output is a JSON manifest with finite train/validation losses,
`iterations` (at most eight), `stop_reason`, `rolled_back`, and
`sample_order_sha256`. A rollback is a valid outcome; this is an API exercise,
not a detector validation or proof of reconstruction benefit.

<!-- reconstruction-minimal:start -->
```python
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
```
<!-- reconstruction-minimal:end -->

The code above is checked byte-for-byte against
[samples/reconstruction_minimal.py](../samples/reconstruction_minimal.py) and
executed by its regression test. The returned `hardened_weights` contains
INT8 arrays; `manifest()` contains diagnostics, not those arrays.
`teacher_forward(input)` returns the reference tensor.
`student_forward(input, weights, activation_drop_probability, generator)`
uses the supplied dequantized weights. The callbacks represent the target
operation only.

For a separately authorized export, the caller must preserve the existing
static initializer topology and validate the complete graph. The immutable
riscy.2 guide had an invalid call signature; see the
[current erratum](MAINTENANCE_ERRATA.md).

If validation does not improve, the engine restores the accepted initial codes.
Do not replace rollback with the nearest re-quantized FP32 weight.

## Data Contract

Split reconstruction data before optimization. Training and held-out
validation must use stable ordering and must remain separate from task
selection/final evaluation. Preserve sample identities and order hashes.

`student_forward` must represent only the named target block and must return
outputs aligned with the teacher. Do not silently expand the target to a
residual, head, or entire model.

## Activation-Drop Mode

`activation_drop_probability` defaults to `0.0`. A QDrop-inspired comparison
may evaluate a predeclared alternative such as `0.5` using held-out
reconstruction loss only. This release does not contain validated YOLO QDrop
evidence.

## Bias Correction

Bias correction is allowed only when the correction folds into an existing
bias initializer without topology change. Record the original bias, correction,
final bias, output mean error, and dtype/accumulator contract.

## Exact Validation Status

Use the phrase **BRECQ-inspired layer-local adaptive rounding infrastructure**.

Validated in the downstream YOLO campaign:

- seven layer-local single-Conv targets;
- deterministic rounding/rollback mechanics;
- ordinary all-S8 Q/DQ export conformance.

Not validated:

- full BRECQ building-block reconstruction;
- residual or C2f block reconstruction;
- whole-head reconstruction;
- QDrop task benefit;
- task-loss block reconstruction;
- end-to-end QAT.

Do not claim “BRECQ implemented” without this qualification.
