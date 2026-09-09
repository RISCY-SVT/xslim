# Release and Provenance

## Identity

Current source: `2.1.2+riscy.2.1` maintenance descendant. Existing-branch source
publication is authorized separately from its locally built candidate assets.
No maintenance tag, binary release or asset upload is authorized or claimed.
Use [source-publication status](SOURCE_PUBLICATION_STATUS.md) and
[certified installation](../INSTALL.md).
The table and release commands below describe the immutable published riscy.2.

| Field | Value |
|---|---|
| Package | `xslim` |
| Version | `2.1.2+riscy.2` |
| Annotated tag | `v2.1.2-riscy.2` |
| Upstream base | `9a33f2f770d00fd02ff8bc0f1907135e9bf47f8c` |
| Upstream base tree | `05d2c8425ab8587abf401fa5976a08d008fdd719` |
| License | Apache-2.0 |

The fork is unofficial and is not published to PyPI.

## Release Assets

The GitHub and GitLab releases contain byte-identical:

```text
xslim wheel
xslim sdist
source archive
documentation archive
SPDX SBOM
release-manifest.json
SHA256SUMS
SHA512SUMS
```

Use `SHA256SUMS` or `SHA512SUMS` to verify downloaded assets. Both remotes must
publish the same names, sizes, and hashes.

## Reproducibility Contract

Release builds fix source commit, `SOURCE_DATE_EPOCH`, Python/build backend
versions, locale, timezone, archive ordering, uid/gid, permissions, and gzip/zip
timestamps. Two clean source exports must produce byte-identical assets.

Reproducible bytes do not prove semantic correctness; the test, typing,
documentation, install, license, and neutrality gates remain separate.

## Included and Excluded Material

Included: source, tests, examples, documentation, license/notices, package
metadata, SBOM, manifest, and checksums.

Excluded: ONNX models, trained weights, predictions, images, datasets,
calibration data, vendor runtimes/binaries, credentials, private paths, core
dumps, and raw research logs.

## Source Provenance

See [UPSTREAM.md](../UPSTREAM.md) for the selected upstream base and
[MODIFICATIONS.md](../MODIFICATIONS.md) for downstream changes. The release
closure did not merge semantic changes from upstream branches.

## Verification

Historical published-release verification follows. For corrected local
maintenance artifacts, use [INSTALL.md](../INSTALL.md), not this old filename.

```bash
sha256sum -c SHA256SUMS
python -m pip install ./xslim-2.1.2+riscy.2-py3-none-any.whl
xslim --version
python -m pip check
```

Expected version is exactly `2.1.2+riscy.2` with no broken requirements.
