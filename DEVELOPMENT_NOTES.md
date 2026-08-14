# XSlim 2.1.2+riscy.2.dev1

This is an unreleased RISCY-SVT development line based on upstream commit
`9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`. It does not modify or supersede
the published `v2.1.2-riscy.1` tag.

The development line adds an opt-in, model-independent constrained asymmetric
INT8 observer, deterministic exact-tensor selectors, post-fusion policy
binding, LSQ qparam locking, and a structural SpacemiT split-model profile
validator. With no enabled range policy, these passes are absent from the
quantization pipeline.

The profile validates graph structure only. It does not prove provider
placement, fusion, kernel selection, correctness on K1X, latency, stability,
or production suitability. No model, weights, calibration data, runtime, or
release artifact is included.
