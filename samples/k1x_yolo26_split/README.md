# K1X YOLO26 six-output split example

This sanitized example records the Stage64-proven graph policy without
including a model, weights, images, or private paths. Confirm that the six
tensor names exist in your own floating-point export before using the config.

Use an independent calibration set. Do not calibrate on the evaluation or
holdout images used to decide accuracy. Populate
`independent_calibration_list.txt` locally; it is intentionally absent from
this repository.

```bash
xslim -c config.json
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

These tools are opt-in diagnostics. They do not make a deployment or accuracy
claim, and the detector contract must be supplied explicitly.
