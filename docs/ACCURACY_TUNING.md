# Accuracy Tuning

Accuracy tuning starts by proving that FP32, preprocessing, decode, and
evaluation agree. Observer sweeps cannot repair a mismatched harness.

## Minimal Diagnostic Order

1. Re-run the FP32 source and accepted split/tail surface through one runner.
2. Verify fixed-fixture shapes, finite values, and output semantics.
3. Audit Q/DQ topology and qparams.
4. Measure task metrics on a calibration-disjoint selection surface.
5. Change one bounded policy at a time.
6. Confirm the winner on an untouched final set.

Expected output is a causal ledger, not merely one mAP number.

If the FP32 baseline differs from its accepted metric, stop. Fix the runner,
preprocessing, decode, annotations, or evaluator before tuning quantization.

## Calibration Data

Use enough natural-distribution examples to cover target classes, object
scales, lighting, and activation ranges. Keep calibration, candidate selection,
and final evaluation disjoint by image ID, file hash, and decoded-pixel hash
where possible.

Vary calibration draw, order, and internal seed only in a predeclared
robustness study. Do not select from the final test set.

## Observer Choices

- `minmax` preserves observed extremes but can waste codes on outliers.
- `percentile` clips tails to improve central resolution.
- `mse` minimizes reconstruction error under candidate qparams.
- `kl` compares deterministic histograms; it is not byte-equivalent to every
  external entropy calibrator.
- constrained MSE enforces zero, floor/headroom, code-count, clipping, and rail
  requirements while optimizing reconstruction.

Use activation proxies only for screening. Task metrics remain the selection
authority.

## Local Policies

Select exact tensors from source/QDQ correspondence evidence. A matching shape
is not enough. Freeze a matched-tensor manifest before reading candidate task
metrics. Fail closed on no match, ambiguity, or conflicting overlap.

Keep legacy `calibration_type` or `max_percentile` settings separate from
`range_policy.enabled=true`; the latter invokes final constrained validation.

## Detector-Specific Reporting

Report:

- mAP50-95, mAP50, and mAP75;
- AP-small/medium/large and AR-small/medium/large;
- per-class AP/AR;
- prediction count and score distribution;
- failures, non-finite values, and score collapse;
- paired image-level uncertainty for close candidates.

A higher mAP candidate can have more false negatives at a chosen score
threshold. Produce TP/FP/FN operating-point tables before deployment.

## Stop Conditions

Reject a candidate on topology drift, QLinear/UINT8, unexpected FP islands,
missing Conv `kernel_shape`, output contract change, non-finite output, score
collapse, or non-deterministic generation.

Do not continue an observer sweep when evidence points to model capacity,
training, or target-runtime arithmetic. Those require a separate method and
authorization.
