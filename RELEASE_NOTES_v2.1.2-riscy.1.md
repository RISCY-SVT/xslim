# XSlim 2.1.2+riscy.1

Unofficial RISCY-SVT K1X/YOLO hardening build. This release is not endorsed by
SpacemiT and is not published to PyPI.

## Source

- Upstream: <https://github.com/spacemit-com/xslim>
- Exact base: `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`
- Upstream version at that base: `2.1.2`
- Fork tag: `v2.1.2-riscy.1`
- Python package version: `2.1.2+riscy.1`

The two-input `ReduceMax` parsing repair is upstream-authored and was already
present at the base commit. RISCY-SVT adds regression coverage and one bounded
empty-reduction identity correction exposed by those tests.

## RISCY-SVT additions

- Expanded `ReduceMax` semantic coverage across opsets, axes, keepdims, noop,
  empty tensors, and Conv-to-ReduceMax behavior.
- Optional `xslim-yolo-output-check` detector-output semantic checker.
- Optional `xslim-qdq-boundary-audit` range and saturation auditor.
- Sanitized K1X/YOLO26 six-output split example with exact letterboxing.
- Apache-2.0 provenance, modifications, and third-party inventories.

## Known limitation

The Stage64 private YOLO26 direct-E2E diagnostic did not match the current
YoloDecode fusion and produced collapsed score channels after quantization.
This release does not claim that direct-E2E route is fixed. The six-output
split remains the proven workflow: bbox/confidence branches remain separate,
only the inference partition is quantized, and post-processing remains float.

## Distribution scope

No model, trained weight, calibration image, COCO image, SpacemiT ONNX Runtime,
or other vendor binary is bundled. The release contains source, wheel, sdist,
SBOM, checksums, and a release manifest only.
