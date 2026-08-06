# Skill Self-Test

- `bash -n`: pass.
- Secret and private-key pattern scan: pass.
- SSH identity resolution: expected GitHub and GitLab identity paths matched;
  no key bytes were read.
- GitHub SSH: authenticated, shell disabled as expected.
- GitLab SSH: authenticated.
- GitHub Actions API: HTTP 403 in both environment and stored modes; preserved.
- GitHub workflow inventory: 0.
- GitHub workflow runs: 0.
- Workflow guard count: 2.
- GitHub default branch and immutable main: pass.
- GitLab user/project/access: `sergey.tyurin`, project 2158, level 40.
- Classification:
  `github-actions-disabled-human-attested-api-unreadable`.

Codex must be restarted before a future session to load the updated skill
automatically.
