# Source hygiene report

Status: pass for the local source/release candidate; publication parity blocked.

- `git diff --check`: pass.
- Tracked symlinks: zero.
- New tracked files above 1 MiB: zero.
- Tracked ONNX/model/checkpoint files: zero.
- Private absolute paths: zero in the release candidate.
- Credential/private-key signatures: zero.
- RPATH/RUNPATH additions: zero.
- Release artifacts contain no full model, weights, calibration data, or ORT.
- `LICENSE` is byte-identical to upstream.
- Banana protected repository and custom executor are read-only controls and
  are rechecked separately at closure.

The full-project mypy invocation is not configured upstream and reports 876
pre-existing PPQ/XSlim errors. The bounded check covering all four new tool
source files passes with no issues; this distinction is retained in evidence.
