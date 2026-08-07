# Stage65A-PUB4 Final Report

## Classification

`stage65a-pub4-github-credential-effective-dual-remote-release-and-ten-asset-parity-actions-disabled-pypi-untouched-publication-closure-pass`

Publication classification: `dual-remote-release-and-asset-parity-pass`.

## Immutable Identities

- Upstream/main: `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c`.
- Clean release: `12647b4a79fe5ec9a3973515a17cece4cb83daf4`.
- Clean release tree: `9ebb21ae2277cf760ed436692c7bc6343ca88d31`.
- Annotated tag object: `604507c45c8eb2ff5c548d30d981203fedc61ea3`.
- Tag peeled commit: `12647b4a79fe5ec9a3973515a17cece4cb83daf4`.
- Payload tree SHA-256: `99b90ede34df19e8ebc54353c94712b3f888f135065f97c1a638b2941c95a893`.
- Stage65A-PUB3 packet tree SHA-256: `60d01ca06a3333a57e4fded2cc10ea3bc7698f9cf54dc83a57fa8935f41c5534`.

All accepted refs and payload bytes matched before release mutation. The clean
release branch, main, and tag were not modified.

## Credential And Actions Gate

The selected GitHub mode was `stored`. Both tested modes resolved to user
`Custler` and repository `RISCY-SVT/xslim`; no environment token variables
were set. The Actions permissions endpoint returned HTTP 200 and
`enabled=false` before draft creation and again after publication.

The repository metadata repair succeeded: Issues are enabled, the description
identifies the unofficial fork, and the homepage points to the immutable tag.

## Releases

- GitHub: https://github.com/RISCY-SVT/xslim/releases/tag/v2.1.2-riscy.1
- GitHub release ID: `366846729`.
- GitLab: https://gitlab.itglobal.com/riscy/sw/xslim/-/releases/v2.1.2-riscy.1
- GitLab project/package IDs: `2158` / `7190`.

The GitHub draft was populated and authenticated-download verified before the
GitLab release was created. GitLab then exposed ten exact package files and
ten release links. Only after those bytes passed did the existing GitHub draft
become public.

## Asset Parity

All ten explicit assets have matching size, SHA-256, SHA-512, and file type
across the local payload, anonymous GitHub downloads, and authenticated GitLab
downloads. Wheel, sdist, and source-archive safety checks passed on all three
surfaces. GitHub-generated source snapshots are excluded from this parity set.

## Publication Safety

Workflow-run counts at 0, 15, 30, 60, and 120 seconds were all zero. The
workflow inventory remained empty. PyPI returned HTTP 404 for
`xslim==2.1.2+riscy.1` at every checkpoint. No PyPI, OIDC, or trusted-publishing
operation was performed.

## Protected Projects

Banana protected main, Stage64, the custom-executor tree, and the accepted
ncnn HEAD/tree/pre-existing three-file diff all remained unchanged. Stage65B
was neither created nor executed.

## Outcome

Proven:

- Effective GitHub credential binding and Actions-disabled API readback.
- Exact immutable branch/tag parity before release.
- Complete GitHub and GitLab release publication on the same tag.
- Ten-asset local/GitHub/GitLab byte parity and archive safety.
- Zero workflow runs and PyPI nonpublication.
- Protected-project invariance.

Broken: none.

Unknown: none within this bounded publication stage.

Human decisions needed: none for publication closure. Stage65B still requires
separate explicit authorization.

Timestamp: `2026-08-07T15:50:23Z`.
