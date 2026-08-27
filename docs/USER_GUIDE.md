# XSlim User Guide

XSlim is an offline ONNX transformation tool. A responsible workflow keeps
model generation, structural validation, task evaluation, and target-runtime
validation as separate evidence surfaces.

## 1. Bind the Source Model

Record the model SHA-256, ONNX opset, inputs, outputs, shapes, dtypes, and any
external-data files before quantization:

```bash
sha256sum model.onnx
python - <<'PY'
import onnx

model = onnx.load("model.onnx", load_external_data=False)
print("opsets", [(item.domain, item.version) for item in model.opset_import])
print("inputs", [item.name for item in model.graph.input])
print("outputs", [item.name for item in model.graph.output])
PY
```

Expected output is a stable identity record. If external data is referenced,
hash every referenced file and keep it beside the ONNX model.

## 2. Prepare Calibration Data

Use representative images that do not overlap the task-selection or final
evaluation surfaces. Keep a deterministic, versioned list. The list alone is
not enough: preprocessing must match production byte-for-byte at the tensor
boundary.

For image input, verify:

- decode library and orientation handling;
- RGB versus BGR;
- resize/crop/letterbox policy and padding value;
- normalization and dtype;
- NCHW versus NHWC;
- dynamic versus fixed dimensions.

An incorrect preprocessing contract can dominate every observer choice.

## 3. Configure and Quantize

Start from [QUICKSTART.md](../QUICKSTART.md), then consult
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md). Run:

```bash
xslim --config config.json
```

Expected outputs are a generated ONNX model and optional analysis reports.
Store the exact config, command, source hash, package version, seed, and data
list hash with each model.

## 4. Validate the Graph

At minimum:

```bash
python - <<'PY'
import onnx

model = onnx.load("output/model_s8_qdq.onnx")
onnx.checker.check_model(model)
onnx.shape_inference.infer_shapes(model)
print("checker and shape inference passed")
PY
```

Then compare graph inputs/outputs, node census, Q/DQ census, initializer dtypes,
and operator attributes with the intended target contract. Use
`xslim-spacemit-profile-check` only when the model is expected to satisfy that
profile.

## 5. Validate Fixed Fixtures

Run the same preprocessed tensors through FP32 and the candidate. Check:

- output names, order, shapes, and dtypes;
- finite values and non-collapsed scores;
- stable output hashes on repeated runs;
- numeric differences and task-level sanity.

Do not require byte equality between different arithmetic backends unless the
contract explicitly promises it.

## 6. Evaluate Task Accuracy

Use the same runner, image list, annotations, threshold, decoder, evaluator,
and serialization for FP32 and candidates. For detection, report mAP, AP by
size/class, AR by size, prediction count, and failures. Use paired image-level
statistics when selecting between close candidates.

H500-like selection surfaces are not final-generalization authority. Preserve
an untouched final set.

## 7. Choose an Operating Point

mAP does not select a production score threshold. Produce TP/FP/FN tables over
the intended IoU, score, maxDets, class, and size ranges. Choose a threshold
from application false-negative and false-positive costs. Record it as part of
the deployed profile.

## 8. Validate the Runtime

On the target device, bind exact model and runtime bytes. Prove provider
placement, fixed-fixture correctness, task transfer, matched performance, and
stability. A structural profile pass is only a prerequisite.

## Artifact Ledger

For every accepted artifact retain:

```text
source model SHA-256
XSlim version and source commit
effective config and command
calibration list and preprocessing identity
generated model SHA-256
structural report
fixed-fixture report
task metrics and prediction identity
target-runtime evidence
known limitations and owner
```
