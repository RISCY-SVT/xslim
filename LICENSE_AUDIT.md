# License audit

## Conclusion

The source and package release candidate is distributable under Apache-2.0
subject to preserving the upstream license and notices. No unresolved bundled
third-party redistribution term was found in the audited tree.

This is an engineering compliance record, not legal advice.

## Evidence

- Upstream declares Apache-2.0 in `pyproject.toml` and GitHub metadata.
- The upstream `LICENSE` SHA-256 is
  `1b2c14e64657882377cc9fdd87d80bdae74c53404ea030876e92864f14406dd6`.
- That file contains the unmodified Apache License 2.0 and its existing
  OpenPPL 2021 and SpacemiT 2023 application notices.
- Upstream commit `9a33f2f...` contains no `NOTICE` file.
- The embedded `src/xslim/ppq_decorator/ppq/` lineage points to OpenPPL PPQ,
  whose public repository declares Apache-2.0. The upstream XSlim license
  already retains the OpenPPL notice.
- Runtime Python dependencies are external requirements and are not vendored
  into the wheel or source distribution. Their observed licenses are listed
  in `THIRD_PARTY_LICENSES.tsv`.

## Distribution checks

The release gate verifies that wheel, sdist, and source archive contain:

- `LICENSE` byte-identical to upstream;
- `UPSTREAM.md` and `MODIFICATIONS.md`;
- `NOTICE-RISCY-SVT` and third-party inventories;
- no trained weights, full models, private paths, credentials, or vendor ORT
  binaries.

## Attribution policy

No new copyright holder is asserted. Git history and upstream notices remain
the source of contributor attribution. The RISCY-SVT notice identifies the
derivative build and its changes without changing license terms.
