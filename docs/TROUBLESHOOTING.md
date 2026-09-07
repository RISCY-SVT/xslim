# Troubleshooting

## Python Support and the Old Reconstruction Example

Use CPython 3.12.3 and the certified dependency constraints in
[INSTALL.md](../INSTALL.md). Python 3.9 conflicts with ONNX 1.21.x;
Python 3.10/3.11 cannot use the accepted NumPy 2.5.2 pin.
An unexpected `fp_weight` keyword comes from the invalid riscy.2 example.
Use the [current executable example](RECONSTRUCTION_GUIDE.md).

## XSlim Cannot Find the Model or Calibration List

Run the command from the directory assumed by the JSON file, or use absolute
paths in a private runtime config. For distributable examples, keep paths
relative and use placeholders.

```bash
test -r model.onnx
test -r calibration.txt
```

Expected: both commands return zero.

## Input Name, Shape, or Dtype Mismatch

Inspect the source model:

```bash
python - <<'PY'
import onnx

model = onnx.load("model.onnx", load_external_data=False)
for value in model.graph.input:
    print(value.name, value.type)
PY
```

Copy exact names and concrete calibration shapes into the config. A shape that
merely looks plausible is not sufficient.

## Static PTQ Rejects Q/DQ Input

Static PTQ expects a floating-point source. Return to the original FP32 export.
Do not remove Q/DQ nodes from a candidate to force it through the pipeline.

## CUDA Falls Back to CPU

Verify `torch.cuda.is_available()` in the active environment. If CUDA is not
available, select `calibration_device: cpu`; do not treat the warning as an
accuracy failure.

## Output Scores Collapse

Stop candidate evaluation. Confirm preprocessing, output ordering, tail/decode,
qparams, finite values, and Q/DQ topology. For YOLO26, use the six-output split
contract; the historical direct-E2E route is not validated.

## SpaceMIT Profile Passes but Board Falls Back

The profile is structural. Bind exact ORT/core/EP libraries on the board and
capture provider assignment. Changed qparams can affect provider compilation
even when graph topology is unchanged.

## Constrained Policy Matches Nothing

With `strict: true`, this is an intentional error. Obtain exact post-fusion
tensor names from a graph/source mapping and regenerate the matched-tensor
manifest. Do not switch strict mode off to hide a stale selector.

## Constraint Is Infeasible

The signed INT8 domain cannot satisfy the requested real range, code budgets,
or clipping/rail limits. Review the physical requirement and observed data.
Do not weaken constraints after reading task results.

## External Data Is Missing

Keep every ONNX external-data file beside the model at its recorded relative
path. Hash and transfer the set together.

## Wheel Installs but CLI Is Missing

Check the active venv and entry points:

```bash
python -m pip show xslim
python -c 'import shutil; print(shutil.which("xslim"))'
```

Reinstall the wheel in a fresh venv if the path points to another environment.

## Release Checksum Fails

Do not install the asset. Download the matching release asset again and compare
its size/hash with both GitHub and GitLab. A filename match is not byte proof.
