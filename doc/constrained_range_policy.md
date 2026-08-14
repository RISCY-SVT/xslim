# Constrained signed-INT8 range policies

`2.1.2+riscy.2.dev1` adds an optional local range policy for signed,
asymmetric, per-tensor activation Q/DQ domains. The selector is explicit and
fails closed. No model-specific tensor name is built into XSlim.

```json
{
  "quantization_parameters": {
    "range_policy_manifest_path": "output/local-policy-manifest.json",
    "custom_setting": [
      {
        "name": "selected-activation-domain",
        "tensor_names": ["exact_tensor_name_from_the_source_graph"],
        "range_policy": {
          "enabled": true,
          "strict": true,
          "objective": "constrained-mse",
          "preserve_zero": true,
          "required_real_min": -0.3,
          "required_real_max": 8.0,
          "semantic_floor": "silu",
          "percentile": 0.9999,
          "search_steps": 32,
          "lock_qparams": true
        }
      }
    ]
  }
}
```

Supported objectives are `minmax`, `percentile`, `mse`, `kl`, and
`constrained-mse`. The search enumerates every signed INT8 zero point, uses
round-to-nearest-even, simulates rail saturation, and applies deterministic
tie handling. `semantic_floor` is opt-in; `"silu"` requests the mathematical
SiLU lower floor only for the selected domain.

`tensor_names` selects exact semantic tensor domains. Existing
`input_names`/`output_names` may instead define a bounded subgraph. Missing
strict matches, ambiguous quantization roots, and incompatible overlaps are
errors. The emitted manifest records every final match and qparam.

Selected qparams can be locked against block-wise LSQ updates and are checked
again after calibration. Terminal Q/DQ nodes remain present: this feature
changes scale, zero point, and represented range, not graph outputs or tail
precision.

The `xslim-spacemit-profile-check` command validates the structural profile
`spacemit_k1x_s8_qdq_split_v1`: signed QDQ, no QLinear or UINT8 zero points,
per-tensor activations, symmetric per-channel Conv weights, explicit Conv
`kernel_shape`, an exact six-output contract, and an optional exact float-tail
hash. Passing the profile is not evidence of runtime placement or speed.
