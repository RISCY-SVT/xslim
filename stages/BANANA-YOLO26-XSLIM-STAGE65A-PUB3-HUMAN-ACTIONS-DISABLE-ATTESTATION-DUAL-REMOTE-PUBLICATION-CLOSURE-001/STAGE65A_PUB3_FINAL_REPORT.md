# Stage65A-PUB3 Final Report

## Classification

`stage65a-pub3-refs-published-github-release-workflows-permission-missing`

Publication classification: exact GitHub/GitLab ref parity completed; release
objects and release assets were not published.

## Outcome

The direct human GitHub web-UI attestation was accepted for this stage only.
Both Actions API credential modes continued to return HTTP 403, so the API
state remains unreadable. Independent corroboration passed: the repository is
the expected true fork, `main` stayed at upstream `9a33f2f...`, workflow
inventory and workflow-run count were both zero, and both release-triggered
jobs in `.github/workflows/publish.yml` have the exact upstream-only guard.

The accepted release artifacts were reused byte-for-byte. A neutral release
manifest and checksum files were generated, and archive safety checks passed.
The old unpublished local tag was replaced with an annotated tag object
`604507c45c8eb2ff5c548d30d981203fedc61ea3`, peeled to the accepted clean
release commit `12647b4a79fe5ec9a3973515a17cece4cb83daf4`.

GitLab `main` was initialized narrowly from upstream. GitHub and GitLab then
received the historical branch, clean release branch, PUB2 evidence branch,
and annotated tag by normal atomic pushes. All published refs matched.

GitHub draft-release creation failed with HTTP 403. A valid diagnostic POST
reported accepted permission alternatives `contents=write` or
`contents=write,workflows=write`. Because the tagged commit changes a workflow
relative to fork `main`, this is classified as missing Workflows permission.
No GitHub release object was created. A GitLab release was deliberately not
created, avoiding asymmetric release surfaces.

## Safety

- GitHub workflow inventory: 0.
- GitHub workflow runs after the failed draft attempt: 0.
- PyPI `xslim==2.1.2+riscy.1`: absent (HTTP 404).
- GitHub releases: 0.
- GitLab releases: 0.
- Clean release source remained at the accepted commit and tree.
- Banana protected refs, custom-executor tree, and `/data/ncnn` state matched
  their accepted identities.
- No package, model, weights, credentials, or private key material is tracked
  on this evidence branch.

## Proven

- The human-attestation fallback is explicit, repository-bound, stage-bound,
  and does not weaken the normal API gate.
- The seven accepted artifact hashes match.
- Main, three accepted branches, annotated tag object, and peeled tag commit
  have exact GitHub/GitLab parity.
- The release-token blocker is distinct from the Actions-disable attestation.

## Broken

- The stored GitHub API credential cannot create a release for this workflow-
  modifying tag because the required Workflows write permission is absent.

## Unknown

- GitHub Actions disabled state cannot be read through the current API token;
  it remains supported by the direct human attestation and corroborating
  evidence only.

## Human Decision

Authorize a GitHub credential with Contents write and Workflows write for
`RISCY-SVT/xslim`, then launch a new bounded publication-resume stage. Do not
recreate or force-update the already published tag.

Timestamp: `2026-08-06T09:31:51Z`
