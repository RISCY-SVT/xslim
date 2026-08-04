# YoloDecode investigation

The exact private Stage64 FP32 source was inspected read-only and was not
copied into this repository or any distribution.

## Observations

- Source graph: 453 nodes; optimized source graph: 383 nodes; direct quantized
  graph: 1,224 nodes.
- No inspected graph contained a `spacemit_functions.YoloDecode` node or
  corresponding `FunctionProto`.
- Ten FP32 source-versus-optimized fixtures had zero output difference.
- The accepted direct-E2E 9a33 artifact still collapsed score channels on all
  100 holdout images.
- The private graph's bbox and confidence paths originate from distinct
  producers. The current matcher requires the expected paths to originate from
  a shared `Split`, so its pattern does not match.
- A public synthetic probe captures that non-match and verifies no semantic
  drift through optimization.

## Decision

No matcher patch was selected. The required proof chain for an implementation
change was incomplete: no narrow public reproducer both matched the private
topology and demonstrated that a patch removed the private score collapse.
The direct-E2E limitation remains explicit. The six-output split remains the
validated workflow.
