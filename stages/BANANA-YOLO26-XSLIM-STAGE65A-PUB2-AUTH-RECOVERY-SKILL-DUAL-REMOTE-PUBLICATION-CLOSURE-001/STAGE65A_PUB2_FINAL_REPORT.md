# Stage65A-PUB2 Final Report

## Classification

`stage65a-pub2-blocked-github-actions-administration-permission-missing-exact-credential-evidence-complete`

Publication classification: `not-published-actions-safety-gate`.

## Result

The accepted clean release source and all six immutable package artifacts
match Stage65A-PUB. A reusable dual-remote authentication skill was installed
and exercised without reading private keys or raw credential configuration.

Git SSH succeeds for `Custler` on GitHub and `sergey.tyurin` on the
self-managed GitLab host. GitLab REST authentication also succeeds for
project `riscy/sw/xslim` (ID 2158) at Maintainer access level 40.

Both GitHub credential modes authenticate as `Custler`, expose admin access to
the exact true fork, and fail the exact Actions Administration GET with HTTP
403. The response identifies `administration=read` as the accepted permission
and reports `Resource not accessible by personal access token`. No environment
override exists; `GH_TOKEN`, `GITHUB_TOKEN`, and `GH_HOST` were unset, so both
modes resolve to the same stored credential.

## Fail-Closed Decision

The stage stopped at the mandatory Actions safety gate. The disable PUT and
readback were not attempted because no credential mode passed the prerequisite
GET. No GitHub metadata, GitLab repository, tag, remote branch, release, or
PyPI state was mutated. The old local unpublished annotated tag remains on
`6765abc...` and must not be published.

The publication evidence branch is local-only and contains sanitized reports.
Its exact commit is recorded by the enclosing result packet after commit.

## Protected State

The clean release branch remains exactly `12647b4...`, tree `9ebb21a...`.
Banana protected main, Stage64, custom executor subtree, and the pre-existing
`/data/ncnn` dirty state retain their accepted identities.

## Required Recovery

Provide a GitHub credential for `Custler` that can read and write repository
Actions Administration for `RISCY-SVT/xslim`. Then rerun the exact GET,
disable PUT, and `enabled=false` readback before any publication mutation.

Restart Codex before the next session so `$k1x_dual_remote_auth` is loaded.
