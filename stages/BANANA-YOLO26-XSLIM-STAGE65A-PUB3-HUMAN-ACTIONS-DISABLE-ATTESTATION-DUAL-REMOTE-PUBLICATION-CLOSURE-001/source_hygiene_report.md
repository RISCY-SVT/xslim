# Source Hygiene Report

Scope: the new Stage65A-PUB3 evidence directory and the immutable release
payload prepared outside Git.

| Check | Result |
|---|---|
| `git diff --check` | pass |
| `git diff --cached --check` | pass |
| Token-prefix and secret patterns | pass, zero matches |
| Authorization/private-token header values | pass, zero matches |
| Private key material | pass, zero matches |
| Raw gh/glab credential paths | pass, zero matches |
| Private home paths | pass, zero matches |
| Symlinks | pass, zero |
| Hardlinks in new evidence | pass, zero |
| Files larger than 1 MiB in new evidence | pass, zero |
| Models, weights, wheels, or archives in Git evidence | pass, zero |
| JSON syntax | pass |
| Wheel/sdist/source archive traversal | pass |
| Wheel/sdist/source archive links/devices/setuid | pass |
| Repository LICENSE/notices retained | pass, unchanged source |

The evidence branch contains reports and manifests only. Release package bytes
remain outside Git. No private key or raw CLI authentication configuration was
read or copied.
