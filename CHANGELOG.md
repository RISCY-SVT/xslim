# Changelog

## 2.1.2+riscy.2 - 2026-08-27

- Added deterministic constrained asymmetric INT8 range policies with strict
  tensor/subgraph selectors and exported qparam manifests.
- Added a fail-closed structural validator for the K1X SpaceMIT signed-S8 Q/DQ
  six-output split contract.
- Added BRECQ-inspired layer-local adaptive rounding infrastructure, held-out
  validation, rollback, and optional bias correction.
- Hardened legacy custom-setting compatibility, qparam locking semantics,
  constrained-range validation, small-array KL, and activation sampling.
- Added detector output, Q/DQ boundary, and profile validation tools.
- Added human onboarding, configuration, K1X YOLO26, reconstruction,
  validation, troubleshooting, limitation, and provenance documentation.
- Added deterministic release packaging, SPDX SBOM, and dual-remote release
  parity evidence.

The release does not include a model, dataset, vendor runtime, or board claim.

## 2.1.2+riscy.1 - 2026-08-07

- Added ReduceMax regression coverage and bounded empty-reduction correction.
- Added opt-in detector-output and Q/DQ boundary tools.
- Added sanitized K1X YOLO26 split examples and downstream provenance.

Upstream history is documented in [UPSTREAM.md](UPSTREAM.md).
