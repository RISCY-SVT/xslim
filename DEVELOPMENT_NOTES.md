# XSlim 2.1.2+riscy.2

This RISCY-SVT downstream release is based on upstream commit
`9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`. It does not modify or supersede
the published `v2.1.2-riscy.1` tag.

The release adds an opt-in, model-independent constrained asymmetric
INT8 observer, deterministic exact-tensor selectors, post-fusion policy
binding, honest locked/unlocked qparam finalization, deterministic adaptive
weight rounding, held-out block reconstruction, bias correction, and a
structural SpacemiT split-model profile validator. Legacy custom settings do
not opt into constrained checks.

Block reconstruction is an explicit library surface. It requires named target
blocks and caller-provided teacher/student adapters, uses deterministic local
random generators, restores the best held-out checkpoint, and rolls back when
validation does not improve. It emits ordinary static INT8 weight values and
does not add training operators, QDQ boundaries, or floating-point islands.
Callers may provide accepted signed-INT8 `initial_codes` as the exact rollback
state. Elements whose accepted code belongs to the FP weight's current
`{floor, ceil}` pair remain trainable; elements outside that pair are frozen at
the accepted code instead of being silently requantized.

The profile validates graph structure only. It does not prove provider
placement, fusion, kernel selection, correctness on K1X, latency, stability,
or production suitability. No model, weights, calibration data, runtime, or
release artifact is included.
