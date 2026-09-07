# K1X YOLO26 Six-Output Cookbook

This cookbook describes the validated graph contract without bundling a model,
dataset, runtime, or private path.

Install the certified Python 3.12.3 environment from [INSTALL.md](../INSTALL.md).
The steps below are a generic graph-contract recipe. They were not executed
as a PTQ campaign in this maintenance.

## Generic Recipe, Stage64 and Frozen B2/C2

`config_stage64_repro.json` is a sanitized historical Stage64 recipe.
`config_accuracy_starting_point.json` is an unvalidated starting proposal.
Neither is an exact B2/C2 reproducer. This distinction also applies to the
generic JSON in the quick start.

Exact historical reproduction requires the accepted FP32 source hash,
original generating source/wheel and full dependency lock, C50 membership
and ordering, seed 65001, exact preprocessing adapter, B2 effective config,
postprocessing/split adapter, six source/output identities, common-tail hash,
and evaluator/annotation/image manifests. C2 additionally requires its frozen
T6 confidence qparam selection manifest and DEV-001B generation adapter.
These project artifacts are not bundled here. Bind them through the Banana
DEV-001A/B/C and Stage65E evidence manifests before asserting reproduction.
The local maintenance package is not a newly certified generator of B2/C2.

## 1. Freeze Input and Output Identity

Record:

```text
input: images, float32, 1x3x640x640
outputs in order:
  P3 bbox
  P3 confidence
  P4 bbox
  P4 confidence
  P5 bbox
  P5 confidence
tail: exact independently hashed float ONNX model
```

Use the actual six tensor names from your verified source graph. Do not infer
names from architecture conventions.

## 2. Match Preprocessing

Use project-exact letterbox preprocessing. The sanitized reference is
[`samples/k1x_yolo26_split/preprocess.py`](../samples/k1x_yolo26_split/preprocess.py).
Verify RGB/BGR, padding, interpolation, scale, stride, NCHW layout, and dtype
against the FP32 runner.

## 3. Prepare Calibration Data

Create a deterministic text list with representative images. Keep it disjoint
from candidate selection and final evaluation. Record the list SHA-256 and
every image identity required by your evidence policy.

```bash
sha256sum independent_calibration_list.txt
```

## 4. Configure Split Boundaries

Start from the sanitized config:

```bash
cp samples/k1x_yolo26_split/config_stage64_repro.json config.json
```

Replace placeholders with your source model, output root, preprocessing
callable, calibration list, and six exact `truncate_var_names`. Keep bbox and
confidence branches separate.

## 5. Quantize

```bash
xslim --config config.json
```

Expected artifacts are a signed-S8 Q/DQ split model and analysis reports. Keep
the exact command, config, package version, seed, source hash, and output hash.

If the exported outputs differ in name, shape, dtype, or order, stop before
tail assembly. Do not repair names ad hoc.

## 6. Validate the SpaceMIT Structure

```bash
xslim-spacemit-profile-check --help
xslim-qdq-boundary-audit --help
```

Require signed INT8 Q/DQ, no QLinear/UINT8/FP16, per-tensor activations,
per-channel symmetric Conv weights, explicit Conv `kernel_shape`, exact graph
census, exact six outputs, and exact float-tail hash.

Structural pass remains separate from board placement.

## 7. Run Host Fixtures

Use deterministic synthetic fixtures and several fixed real images. Run FP32
split+tail and INT8 split+tail through the same preprocessing and decoder.
Require finite `1x300x6`, valid boxes/classes/scores, no score collapse, and
stable hashes per surface.

## 8. Prove Board Placement Separately

Bind exact model and SpaceMIT ORT/runtime bytes on the K1X board. Use provider
profiles/logs to prove one intended partition and zero unexpected CPU inference
regions. The separate CPU float tail is intentional, not fallback.

## 9. Evaluate COCO

Use one exact runner and evaluator for FP32 and candidates. Report mAP/AP/AR,
size bins, per-class metrics, prediction hashes/counts, failures, and paired
uncertainty. Selection and full-val surfaces must remain distinct.

## 10. Select the Operating Threshold

Build TP/FP/FN tables over deployment-relevant score and IoU thresholds. A
higher-AP model can still trade recall for fewer false positives. Record a
model-specific threshold and rollback profile before application use.

B2 remains universal control/rollback. The existing C2 TIER-1 waiver approves
only a separate frozen higher-AP application/research profile. Its historical
universal gate remains FAIL. At score 0.25, IoU 0.50, maxDets 100, C2 has fewer
FP and more FN than B2; choose a C2-specific threshold before any application
default. This guide does not grant a new waiver or promote a runtime.

## Direct-E2E Limitation

YoloDecode source support is present, but the frozen validated YOLO26 split
graphs do not use it. A historical direct-E2E route collapsed scores on
100/100 images. The proven recommendation is six-output split plus exact float
tail.
