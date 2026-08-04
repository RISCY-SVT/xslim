# Stage65A final report

## Classification

`stage65a-blocked-repository-access-local-release-candidate-complete`

The source hardening, tests, deterministic packages, SBOM, license evidence,
and local annotated tag are complete. The required GitHub fork and release are
not published because GitHub rejected both fork attempts with HTTP 403 for the
active token. Creating a non-fork substitute would violate the stage contract.

## Identities

| Item | Identity |
|---|---|
| Upstream | `spacemit-com/xslim` |
| Upstream base | `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c` |
| Upstream tree | `05d2c8425ab8587abf401fa5976a08d008fdd719` |
| Upstream version | `2.1.2` |
| Local branch | `riscy/k1x-yolo26-hardening-001` |
| Release source commit | `6765abc00e9d8dc93e468be0cbc7c188aefdccdb` |
| Release source tree | `795b59ef4d3378d1f2470693cb6d23a40f3ba257` |
| Package version | `2.1.2+riscy.1` |
| Local annotated tag | `v2.1.2-riscy.1` (`b2c2f921...`, peeled to `6765abc...`) |
| Intended fork | `RISCY-SVT/xslim` (not created) |

Upstream `main` remained exactly at the required base. The upstream-authored
two-input ReduceMax repair is retained and attributed to upstream.

## Implemented delta

1. Added expanded ReduceMax regression coverage and fixed the independently
   exposed empty-reduced-domain semantics using ONNX reduction identities.
2. Added opt-in `xslim-yolo-output-check`; it reports malformed, non-finite,
   zero, constant, and low-score outputs and fails closed only when requested.
3. Added opt-in `xslim-qdq-boundary-audit` with Q/DQ range, saturation, bias,
   MAE, normalized MAE, cosine, histogram, and per-image hash evidence.
4. Added a sanitized K1X/YOLO26 six-output split example with exact tensor
   names, project letterbox preprocessing, independent-calibration warning,
   and post-export validation commands.
5. Added Apache provenance, modifications, notices, release notes, package
   metadata, and the explicit unofficial/non-endorsed classification.

## Regression decisions

- Upstream 9a33 passes all three imported Stage64 ReduceMax controls.
- The expanded ReduceMax suite passes opset conversion, multiple/negative
  axes, keepdims, both empty-axes modes, empty float/int domains, empty
  unreduced dimensions, and Conv-to-ReduceMax semantics.
- The direct-E2E private YOLO26 artifact remains rejected: score channels are
  collapsed on 100/100 holdout images.
- The direct graph does not match the current YoloDecode fusion pattern. No
  matcher patch was selected without the required narrow causal proof.
- The six-output split remains the validated route. Stage64's 2.1.1 and 9a33
  deployable split artifacts are byte-identical (`ac855266...12d6c`) and the
  imported 10-image host check has zero score collapses.

## Validation

| Gate | Result |
|---|---|
| Upstream baseline | 122 passed, 7 warnings, 65 subtests |
| Final full pytest | 140 passed, 7 warnings, 65 subtests |
| Ruff | pass |
| Python compile | pass |
| New-tool mypy scope | pass, 4 source files |
| Whole-project mypy | not configured; 876 inherited errors retained |
| Fresh wheel install | pass, three CLI help smokes, `pip check` |
| Fresh sdist install | pass, three CLI help smokes, `pip check` |
| Archive safety/content | pass |
| License/notice payload | pass |

## Distribution candidate

| Artifact | SHA-256 |
|---|---|
| Wheel | `e5c14c60ac6545a3b56f9921cf7fd8a4e191e36085b45b29b2e8a6906eab3116` |
| Deterministic sdist | `2913550e29f46816487109d2954465d67cbe6dd342587830ada94b866f8aaaa9` |
| Source archive | `6d57adcd41ae9a0360fea47c8d8611dd6983a8d0cb974b68f1bcce39bd9da501` |
| SPDX SBOM | `cfdbb343c4e78b7c0aed3cc9311aa4a82e8606afe8edbaf0fdd3fc227d61f801` |

Two clean builds produce byte-identical wheels and normalized sdists. Raw
setuptools sdists contain build-time tar/gzip metadata; their extracted file
content is identical, and only the normalized deterministic sdist is selected.

## License and hygiene

The upstream Apache-2.0 license is unchanged. Applicable PPQ/OpenPPL lineage,
upstream notices, RISCY modification notices, and external dependency inventory
are present. No model, weight, ORT runtime, calibration data, credential,
private path, escaping link, or oversized generated artifact is tracked or
bundled. No PyPI publication was attempted.

## Publication blocker

Both exact authorized commands failed with `HTTP 403: Resource not accessible
by personal access token`. The human must grant/refresh organization fork
permission, create the fork with GitHub's fork mechanism, then push this branch
and tag normally and attach the manifested release payload. Stage65B was not
created or executed.
