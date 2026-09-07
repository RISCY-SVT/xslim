# Validation Tools

The downstream CLIs are opt-in validators. They do not alter a model.

## Documentation Checks

`tools/release/check_docs.py` gives each fenced block a content-bound ID and
separates parsed, compiled, API-bound, executed-synthetic and executed-e2e
states. A syntax pass does not prove an executable workflow. External links
are marked unchecked; internal file targets are checked locally.

```bash
python tools/release/check_docs.py --root . --output docs-check.json --execute-synthetic
```

Only the verbatim reconstruction example is executed by this opt-in command.
Generic PTQ, board, and dataset-dependent recipes keep explicit not-run
reasons. The [erratum](MAINTENANCE_ERRATA.md) explains the earlier counts.

## Detector Output Check

```bash
xslim-yolo-output-check --help
```

Use it on fixed input tensors to check output shape, finite values, boxes,
scores, classes, detection count, score distribution, and collapse guards.
Expected output is a JSON report and a nonzero exit code on a failed guard.

If output columns differ from the assumed layout, supply the correct indices.
Do not reinterpret a custom output tensor as `1x300x6` without proof.

## Q/DQ Boundary Audit

```bash
xslim-qdq-boundary-audit --help
```

The audit binds explicit tensors and reports qparams, representable ranges,
clipping/rail fractions, bias, MAE, normalized MAE, cosine, histograms, and
hashes over supplied fixtures.

If a tensor cannot be mapped uniquely to a Q/DQ site, fix the mapping. Similar
shape is not source correspondence.

## SpaceMIT Structural Profile

```bash
xslim-spacemit-profile-check --help
```

The profile checks signed S8 Q/DQ, qparam/Conv/MatMul contracts, FP16/QLinear
absence, graph census, external data, exact outputs, and optional tail identity.
It fails closed when the structural contract is incomplete.

Provider placement still requires target-runtime profiles/logs.

## Generic ONNX Checks

```bash
python - <<'PY'
import onnx

model = onnx.load("candidate.onnx")
onnx.checker.check_model(model)
onnx.shape_inference.infer_shapes(model)
print("pass")
PY
```

Checker/shape-inference pass is necessary but does not prove quantization
semantics or accuracy.

## Release Checks

```bash
python -m build
twine check dist/*
check-wheel-contents dist/*.whl
python -m pip install dist/*.whl
python -m pip check
```

Release validation should also compare two clean builds byte-for-byte and
verify SPDX SBOM, checksums, fresh wheel/sdist installs, CLI help, and examples.
