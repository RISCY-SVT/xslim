# Skill Self-Test

The helper completed its read-only preflight and emitted narrow classification
`github-actions-administration-permission-missing`.

Validated behavior:

- resolved only SSH `IdentityFile` paths, without reading key bytes;
- recorded credential-variable presence without values;
- tested GitHub environment and stored modes separately;
- tested the explicit self-managed GitLab host in both modes;
- sanitized token lines and credential-config paths;
- recorded GitHub SSH success despite its expected nonzero shell exit;
- detected HTTP 403 independently from successful Git SSH;
- proved GitLab project ID 2158 and access level 40;
- passed shell syntax and secret-pattern scans.

The standard upstream skill validator expects hyphen-case names. This lab's
existing K1X skills and the explicit requested invocation use underscores, so
the installed frontmatter intentionally remains `k1x_dual_remote_auth`.
