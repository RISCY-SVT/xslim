# Configuration Reference

XSlim accepts a JSON file through `xslim --config config.json`. Unknown legacy
fields may be ignored by inherited parsers, so validate effective output and
prefer the documented fields below.

## `model_parameters`

| Field | Type | Purpose |
|---|---|---|
| `onnx_model` | string | Floating-point input ONNX path |
| `working_dir` | string | Output and intermediate directory |
| `output_prefix` | string | Generated artifact prefix |
| `skip_onnxsim` | boolean | Skip ONNX simplification |
| `opset` | integer or null | Convert default ONNX opset before output |

Static PTQ rejects a source graph that already contains `QuantizeLinear` or
`DequantizeLinear`.

## `calibration_parameters`

| Field | Type | Purpose |
|---|---|---|
| `calibration_step` | integer | Number of calibration batches |
| `calibration_batch_size` | integer | Batch size |
| `calibration_device` | `cpu` or `cuda` | Observer execution device |
| `calibration_type` | `default`, `minmax`, `percentile`, `kl`, `mse` | Global observer |
| `input_parameters` | array | One record per model input |

Input fields:

| Field | Type | Purpose |
|---|---|---|
| `input_name` | string | Exact graph input name |
| `input_shape` | integer array | Concrete calibration shape |
| `file_type` | `img`, `npy`, `raw` | Calibration source format |
| `dtype` | string | Tensor dtype, default `float32` |
| `color_format` | `rgb` or `bgr` | Image channel order |
| `mean_value` | number array | Per-channel mean |
| `std_value` | number array | Per-channel divisor |
| `preprocess_file` | string or null | `path.py:function` custom preprocessor |
| `data_list_path` | string | Deterministic input list |

When `preprocess_file` is set, treat that callable as executable code and
review it before use.

## `quantization_parameters`

| Field | Type | Purpose |
|---|---|---|
| `precision_level` | integer | XSlim precision policy |
| `finetune_level` | integer | Inherited finetune policy |
| `analysis_enable` | boolean | Emit graphwise analysis |
| `truncate_var_names` | string array | Exact exported graph boundaries |
| `ignore_op_types` | string array | Legacy operator-type exclusion |
| `ignore_op_names` | string array | Legacy operator-name exclusion |
| `max_percentile` | number or null | Global percentile ceiling |
| `custom_setting` | array or null | Local legacy/constrained settings |
| `range_policy_manifest_path` | string | Required manifest for constrained policies |

Ignoring operators can create floating-point islands and is not compatible
with an all-S8 continuity requirement unless separately validated.

## Local `custom_setting`

Each entry may select exact `tensor_names`, or a bounded contract using both
`input_names` and `output_names`. Optional legacy fields are
`calibration_type`, `max_percentile`, and `precision_level`.

Legacy entries do not opt into constrained-range finalization. Constrained
behavior requires:

```json
{
  "name": "activation-domain-a",
  "tensor_names": ["exact_tensor_name"],
  "range_policy": {
    "enabled": true,
    "strict": true,
    "objective": "constrained-mse",
    "preserve_zero": true,
    "required_real_min": -0.2784645427610738,
    "required_real_max": null,
    "required_intervals": [{
      "name": "silu_negative_trough",
      "real_min": -0.2784645427610738,
      "real_max": 0.0,
      "minimum_codes": 3
    }],
    "minimum_positive_codes": 8,
    "minimum_negative_codes": 3,
    "maximum_clipping_fraction": null,
    "maximum_rail_fraction": null,
    "percentile": 0.9999,
    "search_steps": 32,
    "scale_epsilon": 1e-12,
    "lock_qparams": true
  }
}
```

Strict no-match, ambiguous match, and incompatible overlap fail closed.

## `lock_qparams`

- `true`: observer-selected qparams must survive reconstruction and export
  exactly.
- `false`: reconstruction may retain final scale/zero point, but the complete
  final range contract is revalidated and exported equality is required.

The manifest records initial/final scale and zero point, deltas, lock state,
selection source, reconstruction source, and ONNX equality.

## Range Constraints

Supported constrained policy fields include `preserve_zero`, required real
min/max, semantic floor, required intervals/code counts, positive/negative code
budgets, clipping/rail limits, percentile, search steps, and scale epsilon.

The policy is signed per-tensor INT8 with round-to-nearest-even simulation.
Infeasible constraints are errors, not silently weakened settings.

For inherited fields not repeated here, see [the upstream reference](../doc/configuration.md).
