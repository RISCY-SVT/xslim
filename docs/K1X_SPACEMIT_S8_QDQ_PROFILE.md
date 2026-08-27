# K1X SpaceMIT Signed-S8 Q/DQ Profile

The profile `spacemit_k1x_s8_qdq_split_v1` checks whether an ONNX split model
has the structural form used by the validated K1X vendor lane.

## Run the Validator

```bash
xslim-spacemit-profile-check --help
```

Provide the candidate model, expected B2 graph/census contract, exact six
outputs, and tail identity according to the CLI help. A pass report lists the
checked graph, qparams, Conv attributes, output contract, and structural risk.

If the tool reports dynamic qparams, UINT8, QLinear, FP16, output mismatch, or
external-data failure, reject the model until the source/export contract is
understood. Do not waive structural failures as provider quirks.

## Proven Structural Requirements

- signed INT8 `QuantizeLinear`/`DequantizeLinear` only;
- zero `QLinearConv` and `QLinearMatMul`;
- zero UINT8 zero-point initializers;
- per-tensor activation qparams;
- signed symmetric per-channel Conv weight qparams;
- explicit valid `kernel_shape` on every Conv;
- static initializer-backed qparams at every Q/DQ site;
- no Cast to/from FP16 and no FP16 graph I/O/value-info/initializers;
- expected MatMul activation and weight qparam contract;
- no unexpected custom domains or operators;
- intact external-data references;
- exact source-node/op/QDQ census against the frozen contract;
- exact six output names, shapes, dtypes, and order;
- exact unquantized common tail identity.

## Six-Output Contract

The split order is:

```text
P3 bbox
P3 confidence
P4 bbox
P4 confidence
P5 bbox
P5 confidence
```

The tail consumes those six float boundaries. Quantizing or changing the tail,
reordering outputs, or merging bbox/confidence domains changes the contract.

## What a Pass Does Not Prove

The profile does not prove:

- SpaceMIT provider placement or partition count;
- fusion or provider-internal node ownership;
- kernel selection or arithmetic equivalence;
- latency, memory use, stability, or thermal behavior;
- task accuracy or application operating point.

Those require exact runtime/board evidence. Parser acceptance alone is not a
hardware claim.
