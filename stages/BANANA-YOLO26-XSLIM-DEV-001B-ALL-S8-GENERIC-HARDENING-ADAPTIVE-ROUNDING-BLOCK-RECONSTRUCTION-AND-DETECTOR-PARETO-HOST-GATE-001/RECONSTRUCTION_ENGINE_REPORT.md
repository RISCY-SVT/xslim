# DEV-001B Reconstruction Engine Report

## Foundation

`xslim.reconstruction` provides deterministic, model-independent primitives
for all-S8 targeted reconstruction:

- signed per-channel adaptive weight rounding initialized from FP weights and
  existing scale/zero point;
- differentiable floor/ceil choices with hard deterministic INT8 export;
- normalized output MSE, mean/bias error, optional rank loss and rounding
  regularization;
- held-out validation, early stopping, best-checkpoint restore and rollback;
- deterministic channel-by-spatial-tile activation sampling;
- topology-preserving correction of an existing bias initializer;
- bounded activation-drop modes `0.0` and `0.5`, defaulting to `0.0`.

Teacher outputs are detached before loss construction. Empty output sets and
integer bias initializers fail closed; bias correction cannot silently truncate
a floating correction into an incompatible accumulator domain.

Training-only soft-rounding state is not an ONNX operator and is removed when
the hardened INT8 initializer is emitted. The engine does not add Q/DQ nodes,
floating-point islands, or custom training operators.

## Detector adapter boundary

The generic engine does not contain YOLO tensor names. Exact R0 and R7 target
resolution, terminal-confidence qparam search, and candidate editing live in
the Banana validation adapter. Raw ONNX models and reconstruction caches remain
outside Git.

The detector adapter binds reconstruction to a deterministic FP graph exported
after XSlim's equalization passes. It rejects a target unless hard rounding that
reference with the accepted B2 weight qparams reproduces every accepted B2
weight code after applying the explicit accepted-code baseline. Legal
`{floor, ceil}` elements remain trainable; post-finetune elements outside that
pair and saturated rail elements remain frozen. This makes rollback an exact
return to B2 rather than a silent requantization of the source model.
