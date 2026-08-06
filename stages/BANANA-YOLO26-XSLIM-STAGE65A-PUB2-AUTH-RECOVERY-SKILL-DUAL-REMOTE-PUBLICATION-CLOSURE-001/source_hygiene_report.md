# Source and Evidence Hygiene

- Clean release branch remains byte-identical and clean.
- Evidence branch changes are limited to this stage's text/JSON/TSV reports.
- No package binaries, model, weights, dataset, dependency wheel, token,
  Authorization header, private key, or raw credential configuration is tracked.
- Auth evidence records only expected `~/.ssh` identity paths, never key bytes.
- Token-prefix, key-material, private credential-config path, symlink, hardlink,
  and large-file scans pass for the export candidate.
- The installed skill is under `/data/.codex` and is not exported; only its
  path and SHA-256 identities are reported.
- No Git diff or remote push targeted the clean release branch, Banana repo,
  custom executor, upstream XSlim, or `/data/ncnn`.
