# XSlim Quick Start

This example quantizes a floating-point image model to static INT8 Q/DQ.

Install the certified Python 3.12.3 environment using [INSTALL.md](INSTALL.md).
This is a generic recipe with user-supplied inputs, not an executed maintenance
workflow or frozen B2/C2 reproduction. The dataset-free executable check is
[the synthetic reconstruction example](docs/RECONSTRUCTION_GUIDE.md).

## 1. Prepare Inputs

```text
project/
  model.onnx
  calibration.txt
  config.json
  images/
```

`calibration.txt` contains one path per line:

```text
./images/0001.jpg
./images/0002.jpg
```

Calibration must use the same resize, crop or letterbox, color order, scaling,
mean, and standard deviation as production preprocessing.

## 2. Create the Configuration

```json
{
  "model_parameters": {
    "onnx_model": "./model.onnx",
    "working_dir": "./output",
    "output_prefix": "model_s8_qdq",
    "skip_onnxsim": false
  },
  "calibration_parameters": {
    "calibration_step": 100,
    "calibration_batch_size": 1,
    "calibration_device": "cpu",
    "calibration_type": "default",
    "input_parameters": [{
      "input_name": "images",
      "input_shape": [1, 3, 640, 640],
      "file_type": "img",
      "color_format": "rgb",
      "mean_value": [0, 0, 0],
      "std_value": [1, 1, 1],
      "data_list_path": "./calibration.txt"
    }]
  },
  "quantization_parameters": {
    "precision_level": 0,
    "finetune_level": 1,
    "analysis_enable": true
  }
}
```

## 3. Quantize

```bash
xslim --config config.json
```

Expected output: `output/model_s8_qdq.onnx` and analysis reports. Exact names
can vary with configuration; keep the effective config and command with the
artifact.

If XSlim refuses a model containing `QuantizeLinear` or `DequantizeLinear`, use
the original floating-point source. Static PTQ does not accept an already
quantized graph.

## 4. Validate Structure

```bash
python - <<'PY'
import onnx

model = onnx.load("output/model_s8_qdq.onnx")
onnx.checker.check_model(model)
print("nodes", len(model.graph.node))
print("outputs", [value.name for value in model.graph.output])
PY
```

For the K1X signed-S8 split profile:

```bash
xslim-spacemit-profile-check --help
```

Structural pass is not provider-placement proof.

## 5. Validate Behavior

Run fixed synthetic and real fixtures through FP32 and INT8. Require finite,
non-collapsed outputs, stable shape, valid scores/classes/boxes, and expected
output ordering. Detector users can run:

```bash
xslim-yolo-output-check --help
```

## 6. Measure Task Accuracy

Evaluate FP32 and INT8 with the same image list, preprocessing, decoder,
threshold, evaluator, and serialization. Report aggregate and size/class
metrics. Choose deployment thresholds from application false-positive and
false-negative costs, not from model mAP alone.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for the complete workflow.
