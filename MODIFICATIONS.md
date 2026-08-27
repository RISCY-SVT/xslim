# Modifications

This file records changes relative to upstream commit
`9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`. New files are RISCY-SVT fork
additions; no ownership claim is made over upstream work.

## Modified upstream files

| File | Purpose |
|---|---|
| `.github/workflows/publish.yml` | Fail closed on downstream forks so a fork release cannot invoke the inherited PyPI publishing jobs. |
| `MANIFEST.in` | Include derivative provenance and notice files in source distributions. |
| `README.md` | Identify the unofficial derivative and link provenance. |
| `README_zh.md` | Identify the unofficial derivative and link provenance. |
| `VERSION_NUMBER` | Identify downstream release `2.1.2+riscy.2`; published tag `v2.1.2-riscy.1` remains immutable. |
| `pyproject.toml` | Add fork URLs, upstream attribution, downstream maintainer metadata, opt-in audit/profile CLIs, and source-layout pytest configuration. |
| `src/xslim/optimizer/__init__.py` | Export the downstream local constrained-range passes and observer. |
| `src/xslim/optimizer/observer.py` | Add deterministic constrained asymmetric signed-INT8 histogram observation. |
| `src/xslim/optimizer/refine.py` | Allow exact-tensor range settings without changing legacy bounded-subgraph settings. |
| `src/xslim/optimizer/training.py` | Keep locked constrained qparams immutable and derive block sample order from local deterministic seeds. |
| `src/xslim/quantizer/xslim.py` | Insert opt-in post-fusion binding and post-calibration verification passes. |
| `src/xslim/xslim_setting.py` | Add strict model-independent selector and constrained-range configuration fields. |
| `src/xslim/ppq_decorator/ppq/executor/op/torch/default.py` | Preserve ONNX `ReduceMax` identity behavior when a selected reduction domain is empty. |
| `tests/test_packaging_standards.py` | Validate the derivative license payload and added console entry points structurally. |
| `tests/test_riscy_release_metadata.py` | Keep the immutable release identity separate from the downstream development version and evidence. |

Each text/code file above carries a prominent RISCY-SVT modification notice,
except `VERSION_NUMBER`, whose parser requires a single PEP 440 version line.
Its `+riscy.2` suffix is the prominent modification identification.

## Added files

- Apache/provenance documents: `UPSTREAM.md`, `MODIFICATIONS.md`,
  `NOTICE-RISCY-SVT`, `LICENSE_AUDIT.md`, `THIRD_PARTY_NOTICES.md`, and
  `THIRD_PARTY_LICENSES.tsv`.
- Unofficial release notes in `RELEASE_NOTES.md`.
- Optional validation tools under `src/xslim/tools/`.
- Regression tests for ReduceMax, detector output semantics, Q/DQ boundaries,
  packaging, and CLI behavior.
- Generic constrained asymmetric INT8 range search, strict local selector,
  post-fusion binding/finalization, and property/integration regressions.
- Generic deterministic channel/spatial sampling, adaptive weight rounding,
  held-out block reconstruction, validation rollback, and bias correction.
- A structural `spacemit_k1x_s8_qdq_split_v1` profile validator; it does not
  make provider-placement or performance claims.
- Human documentation, development notes, and schema in `DEVELOPMENT_NOTES.md`,
  `doc/constrained_range_policy.md`, and
  `doc/constrained_range_policy.schema.json`.
- Two explicitly separated sanitized K1X/YOLO26 six-output split configs under
  `samples/k1x_yolo26_split/`: the exact Stage64 reproduction policy and an
  unvalidated vendor accuracy-tuning starting point.
- Release provenance, workflow-safety, and sample-claim regression tests.

## Explicit non-changes

- The upstream two-input `ReduceMax` repair remains unchanged and attributed
  to upstream.
- No model, trained weight, calibration image, COCO image, ORT runtime, or
  private Stage64 artifact is distributed.
- No unconditional model-specific rule was added to XSlim quantization.
- No YoloDecode matcher change was selected for this release.
- The tagged release source contains no Stage evidence or raw lab artifacts.
- The published `v2.1.2-riscy.1` tag and package remain unchanged. This source
  prepares the separate `v2.1.2-riscy.2` downstream release.
