<!-- Modified by RISCY-SVT in 2026 to document the unofficial downstream release. -->
> **Unofficial RISCY-SVT downstream.** This fork is based on XSlim 2.1.2 source
> commit `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`. It is not endorsed by
> SpacemiT and is not published to PyPI. See [UPSTREAM.md](UPSTREAM.md).

# XSlim 2.1.2+riscy.2.1 (Local Maintenance)

[English](README.md) | [Русский](docs/ru/README.md) | [中文](README_zh.md)

XSlim converts floating-point ONNX models to static INT8 Q/DQ, dynamic INT8,
or FP16 models. The RISCY-SVT release adds deterministic constrained range
policies, layer-local adaptive rounding infrastructure, and structural checks
for the K1X SpaceMIT signed-S8 split-model contract.

## Install

Use the locally built maintenance wheel and certified dependency constraints.
The executed environment is CPython 3.12.3 on Ubuntu 24.04 Linux x86_64 (CPU).
This maintenance is not pushed, tagged or released; published riscy.2 remains
the frozen model-generation reference. See [INSTALL.md](INSTALL.md).

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --constraint requirements-certified-python312.txt --extra-index-url https://download.pytorch.org/whl/cpu ./xslim-2.1.2+riscy.2.1-py3-none-any.whl
xslim --version
```

Expected output:

```text
xslim 2.1.2+riscy.2.1
```

The package is intentionally absent from PyPI. If `pip install xslim` returns
another build, uninstall it and install the downloaded downstream wheel by
path. Full installation options are in [INSTALL.md](INSTALL.md).

For a checked-out source tree:

```bash
python -m pip install .
python -m pip install -e .
python -m build
```

## Minimal PTQ Run

Prepare a floating-point ONNX model, a text file with one calibration image per
line, and `config.json`:

This is a generic recipe, not frozen B2/C2 reproduction. Model generation was
not executed in this documentation maintenance. For a runnable dataset-free
API exercise, use [the reconstruction example](docs/RECONSTRUCTION_GUIDE.md).

```json
{
  "model_parameters": {
    "onnx_model": "./model.onnx",
    "working_dir": "./output",
    "output_prefix": "model_s8_qdq"
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

Run:

```bash
xslim --config config.json
python -m xslim --config config.json
```

Expected output is an ONNX model under `working_dir` plus analysis files when
`analysis_enable` is true. Always run `onnx.checker`, a fixed-fixture semantic
test, and task accuracy evaluation before deployment.

If XSlim reports a missing calibration file or input mismatch, verify every
path, input name, shape, dtype, color order, normalization constant, and
letterbox rule against the original model. See [QUICKSTART.md](QUICKSTART.md).

## Command-Line Modes

```bash
# Static PTQ from JSON
xslim --config config.json

# Dynamic INT8
xslim -i model.onnx -o model.dynamic.onnx --dynq

# FP16 conversion
xslim -i model.onnx -o model.fp16.onnx --fp16

# ONNX simplification only
xslim -i model.onnx -o model.slim.onnx

# Structural and semantic validators
xslim-spacemit-profile-check --help
xslim-yolo-output-check --help
xslim-qdq-boundary-audit --help
```

`xslim-spacemit-profile-check` proves only graph structure. It does not prove
SpaceMIT provider placement, kernel selection, latency, or stability.

## K1X YOLO26 Status

The proven K1X workflow uses:

- signed INT8 Q/DQ, with no QLinear or UINT8 qparams;
- per-tensor activation scales and symmetric per-channel Conv weight scales;
- explicit `kernel_shape` on every Conv;
- six ordered bbox/confidence outputs;
- a separate exact floating-point CPU tail.

XSlim contains YoloDecode source support, but the frozen validated YOLO26 split
graphs do not use it. A historical direct-E2E YOLO26 route collapsed scores on
100/100 images. The current recommendation is the six-output split plus exact
float tail. Follow [the K1X YOLO26 cookbook](docs/K1X_YOLO26_COOKBOOK.md).

## Reconstruction Scope

This release provides **BRECQ-inspired layer-local adaptive rounding
infrastructure**: deterministic soft-to-hard weight rounding, held-out
validation, best-checkpoint restore, rollback, and optional bias correction.

The validated YOLO campaign used seven layer-local single-Conv targets. Full
BRECQ building-block reconstruction, residual/C2f reconstruction, whole-head
reconstruction, QDrop, task-loss reconstruction, and QAT are not validated.
See [RECONSTRUCTION_GUIDE.md](docs/RECONSTRUCTION_GUIDE.md).

## Documentation

- [Current maintenance errata and verification limits](docs/MAINTENANCE_ERRATA.md)
- [Installation](INSTALL.md)
- [Quick start](QUICKSTART.md)
- [User guide](docs/USER_GUIDE.md)
- [Configuration reference](docs/CONFIG_REFERENCE.md)
- [Accuracy tuning](docs/ACCURACY_TUNING.md)
- [K1X SpaceMIT S8-QDQ profile](docs/K1X_SPACEMIT_S8_QDQ_PROFILE.md)
- [K1X YOLO26 cookbook](docs/K1X_YOLO26_COOKBOOK.md)
- [Validation tools](docs/VALIDATION_TOOLS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Limitations](docs/LIMITATIONS.md)
- [Release and provenance](docs/RELEASE_AND_PROVENANCE.md)

## Release Boundaries

The immutable riscy.2 release includes source, documentation, wheel, sdist, SPDX SBOM, manifest,
and checksums. It includes no model, weight, prediction, dataset, SpaceMIT
runtime, or vendor binary. Models produced with XSlim require independent
accuracy, placement, performance, and stability validation.

B2 remains the universal vendor control and rollback. C2 remains the exact
frozen higher-AP profile under the existing TIER-1 waiver; its historical
universal recall gate remains FAIL. A C2-specific score threshold must be
selected before application-default use. Maintenance does not enact a default.

## License and Support

XSlim is Apache-2.0 licensed. Downstream modifications and third-party notices
are documented in [MODIFICATIONS.md](MODIFICATIONS.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Use [SUPPORT.md](SUPPORT.md) for support boundaries and [SECURITY.md](SECURITY.md)
for private security reporting.
