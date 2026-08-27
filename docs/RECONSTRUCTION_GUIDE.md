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

## Minimal Pattern

```python
from xslim.reconstruction import ReconstructionConfig, reconstruct_block

config = ReconstructionConfig(
    seed=65001,
    max_iterations=200,
    validation_interval=10,
    patience=5,
)

result = reconstruct_block(
    block_name="conv-target",
    fp_weight=fp_weight,
    scale=weight_scale,
    zero_point=weight_zero_point,
    train_inputs=train_inputs,
    train_teacher_outputs=train_teacher_outputs,
    validation_inputs=validation_inputs,
    validation_teacher_outputs=validation_teacher_outputs,
    student_forward=student_forward,
    config=config,
    initial_codes=accepted_int8_codes,
)
```

Expected result: hardened INT8 codes plus train/validation loss, sample-order
hash, best iteration, stop reason, and rollback status. The caller must export
the codes through an existing static weight initializer and revalidate the
full graph.

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
