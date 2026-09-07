# XSlim Release Notes

## 2.1.2+riscy.2.1: Local Maintenance, Not Published

D1 corrects and executes the reconstruction example. D2 narrows metadata to
Python >=3.12.3,<3.13 and preserves the certified numeric dependency closure.
D3 separates documentation parsing, API binding and execution and explains
generic versus exact frozen B2/C2 reproduction. See
[the current erratum](docs/MAINTENANCE_ERRATA.md).
No runtime Python module, model, qparam or scientific result changes. No new
tag or release was created. The following riscy.2 record remains historical.

## XSlim 2.1.2+riscy.2 (Published)

Unofficial RISCY-SVT downstream release for reproducible ONNX PTQ and K1X
SpaceMIT signed-S8 Q/DQ validation. It is not endorsed by SpacemiT and is not
published to PyPI.

## Identity

- Tag: `v2.1.2-riscy.2`
- Package: `2.1.2+riscy.2`
- Upstream: <https://github.com/spacemit-com/xslim>
- Upstream base commit: `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`
- Upstream base tree: `05d2c8425ab8587abf401fa5976a08d008fdd719`

## Added downstream capabilities

- Deterministic, fail-closed exact-tensor and bounded-subgraph local policies.
- Constrained asymmetric signed-INT8 range search with explicit representable
  intervals, code budgets, clipping limits, and exported-qparam verification.
- Honest `lock_qparams=true/false` behavior through reconstruction.
- Deterministic stratified activation sampling and small-array histogram KL.
- BRECQ-inspired layer-local adaptive rounding infrastructure with held-out
  validation, best-checkpoint restore, rollback, and optional bias correction.
- Structural validator for the `spacemit_k1x_s8_qdq_split_v1` profile.
- Optional YOLO output and Q/DQ boundary audit commands.
- Human installation, configuration, reconstruction, K1X profile, YOLO26,
  troubleshooting, validation, provenance, and Russian quick-start guides.

## Validated scope and limits

The validated downstream YOLO campaign used seven layer-local single-Conv
targets. Full BRECQ building-block reconstruction, residual/C2f
reconstruction, whole-head reconstruction, QDrop, task-loss reconstruction,
and QAT are not validated.

XSlim includes YoloDecode source support, but the frozen validated YOLO26
graphs use six ordered bbox/confidence outputs followed by an exact float CPU
tail. A historical direct-E2E route collapsed scores on 100/100 images. The
current recommendation is the split contract, not direct E2E.

Passing the structural SpaceMIT profile does not prove provider placement,
kernel selection, numerical equivalence, accuracy, latency, stability, or
production suitability. Generated models require separate host and target
validation.

The sanitized `config_stage64_repro.json` remains a reproducibility reference.
`config_accuracy_starting_point.json` remains guidance and has not passed K1X board or COCO validation as a standalone released configuration.

## Distribution boundary

Release assets contain source, documentation, wheel, sdist, SPDX SBOM,
manifest, and checksums. They contain no ONNX model, weight, prediction,
dataset, SpaceMIT runtime, vendor binary, credential, or private lab path.

The inherited upstream PyPI workflow is fail-closed on this fork. This release
is published only as GitHub and GitLab release assets.
