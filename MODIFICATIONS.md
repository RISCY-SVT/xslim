# Modifications

This file records changes relative to upstream commit
`9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`. New files are RISCY-SVT fork
additions; no ownership claim is made over upstream work.

## Modified upstream files

| File | Purpose |
|---|---|
| `MANIFEST.in` | Include derivative provenance and notice files in source distributions. |
| `README.md` | Identify the unofficial derivative and link provenance. |
| `README_zh.md` | Identify the unofficial derivative and link provenance. |
| `VERSION_NUMBER` | Apply the PEP 440 local version `2.1.2+riscy.1`; the `riscy` suffix is the file's modification marker. |
| `pyproject.toml` | Add fork URLs, opt-in audit CLIs, and source-layout pytest configuration. |
| `src/xslim/ppq_decorator/ppq/executor/op/torch/default.py` | Preserve ONNX `ReduceMax` identity behavior when a selected reduction domain is empty. |
| `tests/test_packaging_standards.py` | Validate the derivative license payload and added console entry points structurally. |

Each text/code file above carries a prominent RISCY-SVT modification notice,
except `VERSION_NUMBER`, whose parser requires a single PEP 440 version line.
Its `+riscy.1` suffix is the prominent modification identification.

## Added files

- Apache/provenance documents: `UPSTREAM.md`, `MODIFICATIONS.md`,
  `NOTICE-RISCY-SVT`, `LICENSE_AUDIT.md`, `THIRD_PARTY_NOTICES.md`, and
  `THIRD_PARTY_LICENSES.tsv`.
- Unofficial release notes in `RELEASE_NOTES_v2.1.2-riscy.1.md`.
- Optional validation tools under `src/xslim/tools/`.
- Regression tests for ReduceMax, detector output semantics, Q/DQ boundaries,
  packaging, and CLI behavior.
- A sanitized K1X/YOLO26 six-output split example under
  `samples/k1x_yolo26_split/`.
- Stage 65A engineering evidence under `stages/`.

## Explicit non-changes

- The upstream two-input `ReduceMax` repair remains unchanged and attributed
  to upstream.
- No model, trained weight, calibration image, COCO image, ORT runtime, or
  private Stage64 artifact is distributed.
- No unconditional model-specific rule was added to XSlim quantization.
- No YoloDecode matcher change was selected for this release.
