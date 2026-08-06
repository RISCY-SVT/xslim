# Human Decision Required

Provision or authorize a GitHub credential for account `Custler` with
repository Administration read/write permission on `RISCY-SVT/xslim`.

Required recovery proof before publication:

```bash
gh api -i repos/RISCY-SVT/xslim/actions/permissions
printf '{"enabled":false}' | \
  gh api --method PUT repos/RISCY-SVT/xslim/actions/permissions --input -
gh api repos/RISCY-SVT/xslim/actions/permissions --jq '.enabled'
```

The first call must return HTTP 200, the PUT HTTP 204, and the readback
`false`. Do not publish the current old local tag; recreate it on
`12647b4a79fe5ec9a3973515a17cece4cb83daf4` only after that safety gate.

Restart Codex before the next session so `$k1x_dual_remote_auth` is loaded.
