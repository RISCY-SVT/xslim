# DEV-001B Generic Hardening Report

Stage: `BANANA-YOLO26-XSLIM-DEV-001B-ALL-S8-GENERIC-HARDENING-ADAPTIVE-ROUNDING-BLOCK-RECONSTRUCTION-AND-DETECTOR-PARETO-HOST-GATE-001`

## Scope

The implementation is model-independent. It preserves legacy `custom_setting`
behavior unless `range_policy.enabled=true` is explicitly selected.

## Implemented contracts

- Legacy calibration type, percentile, precision, per-channel, FP32 and
  no-match behavior remain outside constrained finalization.
- Constrained and legacy entries can coexist; contradictory constrained
  ownership remains fail-closed and configuration order is semantic-neutral.
- `lock_qparams=true` restores observer-selected qparams after reconstruction.
- `lock_qparams=false` preserves reconstructed qparams and validates their full
  final contract before export.
- Final validation covers finite positive scale, signed INT8 zero point,
  representable zero, required bounds and intervals, positive/negative code
  budgets, clipping/rail limits, and exported initializer equality.
- Small-array KL uses observed multiplicity instead of equal-point weights.
- Deterministic activation sampling balances the first draw across channels
  and spatial tiles, including budgets smaller than the full Cartesian set,
  and emits a selected-index SHA-256. The v2 implementation vectorizes
  detector-scale candidate ranking instead of materializing one Python object
  per activation element.
- The SpacemiT structural profile rejects FP16 values/Casts, dynamic qparams,
  unsigned qparams, unexpected domains, invalid external data, MatMul qparam
  violations, Conv kernel mismatches, output/tail drift, and reference-census
  drift.
- Block training uses an explicit local RNG, block-name-derived seeds that are
  independent of block traversal order, stable sample-order hashes, held-out
  validation, early stopping, best-checkpoint restore and rollback. The
  no-seed compatibility path clones the already-advanced process RNG state and
  never mutates it.

## Limits

The structural profile does not prove provider placement, kernel choice,
latency, stability, or provider-internal arithmetic. No board command is part
of this stage.
