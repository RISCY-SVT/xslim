# K1X YOLO26 six-output split example

This sanitized example contains two deliberately separate policies without a
model, weights, images, or private paths. Confirm that the six tensor names
exist in your own floating-point export before using either config.

Use an independent calibration set. Do not calibrate on the evaluation or
holdout images used to decide accuracy. Populate
the referenced calibration list locally; it is intentionally absent from this
repository.

`config_stage64_repro.json` reproduces the policy that Stage64 validated on
host and K1X: 50 calibration images, `precision_level=0`, and
`finetune_level=1` with project-exact preprocessing.

```bash
xslim -c config_stage64_repro.json
```

`config_accuracy_starting_point.json` follows current vendor accuracy-tuning
guidance: an independent corpus of at least 500 images,
`precision_level=1`, and `finetune_level=2`. It is a starting point for a new
accuracy study. It has not passed K1X board or COCO validation in this release.

```bash
xslim -c config_accuracy_starting_point.json
```

The six bbox and confidence boundaries remain separate. XSlim quantizes the
inference partition; the post-processing partition remains floating point.
Do not concatenate branch families before per-tensor quantization.

After export, run the semantic gate against the recombined `1x300x6` output:

```bash
xslim-yolo-output-check \
  --model recombined.onnx \
  --image-list independent_holdout.txt \
  --preprocess preprocess.py:preprocess_one \
  --output-name output0 \
  --score-column 4 \
  --class-column 5 \
  --expected-shape 1,300,6 \
  --score-floor 0 \
  --fail-on-violation \
  --report semantic-report.json
```

Audit selected Q/DQ boundaries independently:

```bash
xslim-qdq-boundary-audit \
  --float-model float-inference.onnx \
  --quant-model quantized-inference.onnx \
  --tensor-list boundaries.txt \
  --image-list independent_holdout.txt \
  --preprocess preprocess.py:preprocess_one \
  --report qdq-boundaries.tsv
```

These tools are opt-in diagnostics. Neither config includes a model, weights,
images, or private paths. They do not make a deployment or accuracy claim, and
the detector contract must be supplied explicitly.
